#!/usr/bin/python

from __future__ import print_function
from bcc import BPF
from bcc.utils import printb

import ctypes as ct
import chardet
import argparse

MAX_MSG_SIZE = 30720
CHUNK_LIMIT = 4

# define BPF program
bpf_text = """
#include <linux/in6.h>
#include <linux/net.h>
#include <linux/socket.h>
#include <net/inet_sock.h>

// Defines

#define socklen_t size_t

// Data buffer message size. BPF can submit at most this amount of data to a perf buffer.
// Kernel size limit is 32KiB. See https://github.com/iovisor/bcc/issues/2519 for more details.
#define MAX_MSG_SIZE 30720  // 30KiB

// This defines how many chunks a perf_submit can support.
// This applies to messages that are over MAX_MSG_SIZE,
// and effectively makes the maximum message size to be CHUNK_LIMIT*MAX_MSG_SIZE.
#define CHUNK_LIMIT 4

enum traffic_direction_t {
    kEgress,
    kIngress,
};

// Structs

// A struct representing a unique ID that is composed of the pid, the file
// descriptor and the creation time of the struct.
struct conn_id_t {
    // Process ID
    uint32_t pid;
    // The file descriptor to the opened network connection.
    int32_t fd;
    // Timestamp at the initialization of the struct.
    uint64_t tsid;
};

// This struct contains information collected when a connection is established,
// via an accept4() syscall.
struct conn_info_t {
    // Connection identifier.
    struct conn_id_t conn_id;

    // The number of bytes written/read on this connection.
    int64_t wr_bytes;
    int64_t rd_bytes;

    // A flag indicating we identified the connection as HTTP.
    bool is_http;
};

// An helper struct that hold the addr argument of the syscall.
struct accept_args_t {
    struct sockaddr_in* addr;
};

// An helper struct to cache input argument of read/write syscalls between the
// entry hook and the exit hook.
struct data_args_t {
    int32_t fd;
    const char* buf;
};

// An helper struct that hold the input arguments of the close syscall.
struct close_args_t {
    int32_t fd;
};

// A struct describing the event that we send to the user mode upon a new connection.
struct socket_open_event_t {
    // The time of the event.
    uint64_t timestamp_ns;
    // A unique ID for the connection.
    struct conn_id_t conn_id;
    // The address of the client.
    struct sockaddr_in addr;
};

// Struct describing the close event being sent to the user mode.
struct socket_close_event_t {
    // Timestamp of the close syscall
    uint64_t timestamp_ns;
    // The unique ID of the connection
    struct conn_id_t conn_id;
    // Total number of bytes written on that connection
    int64_t wr_bytes;
    // Total number of bytes read on that connection
    int64_t rd_bytes;
};

struct socket_data_event_t {
  // We split attributes into a separate struct, because BPF gets upset if you do lots of
  // size arithmetic. This makes it so that it's attributes followed by message.
  struct attr_t {
    // The timestamp when syscall completed (return probe was triggered).
    uint64_t timestamp_ns;

    uint32_t pid;

    // The type of the actual data that the msg field encodes, which is used by the caller
    // to determine how to interpret the data.
    enum traffic_direction_t direction;

	// The size of the original message. We use this to truncate msg field to minimize the amount
    // of data being transferred.
    uint32_t msg_size;
  } attr;
  char msg[MAX_MSG_SIZE];
};

// Maps

// A map of the active connections. The name of the map is conn_info_map
// the key is of type uint64_t, the value is of type struct conn_info_t,
// and the map won't be bigger than 128KB.
BPF_HASH(conn_info_map, uint64_t, struct conn_info_t, 131072);
// An helper map that will help us cache the input arguments of the accept syscall
// between the entry hook and the return hook.
BPF_HASH(active_accept_args_map, uint64_t, struct accept_args_t);
// Perf buffer to send to the user-mode the data events.
BPF_PERF_OUTPUT(socket_data_events);
// A perf buffer that allows us send events from kernel to user mode.
// This perf buffer is dedicated for special type of events - open events.
BPF_PERF_OUTPUT(socket_open_events);
// Perf buffer to send to the user-mode the close events.
BPF_PERF_OUTPUT(socket_close_events);
BPF_PERCPU_ARRAY(socket_data_event_buffer_heap, struct socket_data_event_t, 1);
BPF_HASH(active_write_args_map, uint64_t, struct data_args_t);
// Helper map to store read syscall arguments between entry and exit hooks.
BPF_HASH(active_read_args_map, uint64_t, struct data_args_t);
// An helper map to store close syscall arguments between entry and exit syscalls.
BPF_HASH(active_close_args_map, uint64_t, struct close_args_t);

// Helper functions

static inline __attribute__((__always_inline__)) bool is_http_connection(struct conn_info_t* conn_info, const char* buf, size_t count) {
    // If the connection was already identified as HTTP connection, no need to re-check it.
    if (conn_info->is_http) {
        return true;
    }

    // The minimum length of http request or response.
    if (count < 16) {
        return false;
    }

    bool res = false;
    if (buf[0] == 'H' && buf[1] == 'T' && buf[2] == 'T' && buf[3] == 'P') {
        res = true;
    }
    if (buf[0] == 'G' && buf[1] == 'E' && buf[2] == 'T') {
        res = true;
    }
    if (buf[0] == 'P' && buf[1] == 'O' && buf[2] == 'S' && buf[3] == 'T') {
        res = true;
    }

    if (res) {
        conn_info->is_http = true;
    }

    return res;
}

static __inline void perf_submit_buf(struct pt_regs* ctx, const enum traffic_direction_t direction,
                                     const char* buf, size_t buf_size, size_t offset,
                                     struct socket_data_event_t* event) {

    // Note that buf_size_minus_1 will be positive due to the if-statement above.
    size_t buf_size_minus_1 = buf_size - 1;

    // Clang is too smart for us, and tries to remove some of the obvious hints we are leaving for the
    // BPF verifier. So we add this NOP volatile statement, so clang can't optimize away some of our
    // if-statements below.
    // By telling clang that buf_size_minus_1 is both an input and output to some black box assembly
    // code, clang has to discard any assumptions on what values this variable can take.
    asm volatile("" : "+r"(buf_size_minus_1) :);

    buf_size = buf_size_minus_1 + 1;

    // 4.14 kernels reject bpf_probe_read with size that they may think is zero.
    // Without the if statement, it somehow can't reason that the bpf_probe_read is non-zero.
    size_t amount_copied = 0;
    if (buf_size_minus_1 < MAX_MSG_SIZE) {
        bpf_probe_read(&event->msg, buf_size, buf);
        amount_copied = buf_size;
    } else {
        bpf_probe_read(&event->msg, MAX_MSG_SIZE, buf);
        amount_copied = MAX_MSG_SIZE;
    }

    // If-statement is redundant, but is required to keep the 4.14 verifier happy.
    if (amount_copied > 0) {
        event->attr.msg_size = amount_copied;
        socket_data_events.perf_submit(ctx, event, sizeof(event->attr) + amount_copied);
    }
}

static __inline void perf_submit_wrapper(struct pt_regs* ctx,
                                         const enum traffic_direction_t direction, const char* buf,
                                         const size_t buf_size, struct socket_data_event_t* event) {
    int bytes_sent = 0;
    unsigned int i;
#pragma unroll
    for (i = 0; i < CHUNK_LIMIT; ++i) {
        const int bytes_remaining = buf_size - bytes_sent;
        const size_t current_size = (bytes_remaining > MAX_MSG_SIZE && (i != CHUNK_LIMIT - 1)) ? MAX_MSG_SIZE : bytes_remaining;
        perf_submit_buf(ctx, direction, buf + bytes_sent, current_size, bytes_sent, event);
        bytes_sent += current_size;
        if (buf_size == bytes_sent) {
            return;
        }
    }
}

static inline __attribute__((__always_inline__)) void process_data(struct pt_regs* ctx, uint64_t id,
                                                                   enum traffic_direction_t direction,
                                                                   const struct data_args_t* args, ssize_t bytes_count) {
    // Always check access to pointer before accessing them.
    if (args->buf == NULL) {
        return;
    }

    // For read and write syscall, the return code is the number of bytes written or read, so zero means nothing
    // was written or read, and negative means that the syscall failed. Anyhow, we have nothing to do with that syscall.
    if (bytes_count <= 0) {
        return;
    }

    uint32_t pid = id >> 32;

    uint32_t kZero = 0;
    struct socket_data_event_t* event = socket_data_event_buffer_heap.lookup(&kZero);
    if (event == NULL) {
        return;
    }

    // Fill the metadata of the data event.
    event->attr.timestamp_ns = bpf_ktime_get_ns();
    event->attr.direction = direction;
    event->attr.pid = id;

    perf_submit_wrapper(ctx, direction, args->buf, bytes_count, event);
}

// Hooks
// original signature: ssize_t read(int fd, void *buf, size_t count);
int kprobe__sys_read(struct pt_regs* ctx, int fd, char* buf, size_t count) {
    uint64_t id = bpf_get_current_pid_tgid();

    //FILTER_PID

    // Stash arguments.
    struct data_args_t read_args = {};
    read_args.fd = fd;
    read_args.buf = buf;
    active_read_args_map.update(&id, &read_args);

    return 0;
}

int kretprobe__sys_read(struct pt_regs* ctx) {
    uint64_t id = bpf_get_current_pid_tgid();

    //FILTER_PID

    // The return code the syscall is the number of bytes read as well.
    ssize_t bytes_count = PT_REGS_RC(ctx);
    struct data_args_t* read_args = active_read_args_map.lookup(&id);
    if (read_args != NULL) {
        // kIngress is an enum value that let's the process_data function
        // to know whether the input buffer is incoming or outgoing.
        process_data(ctx, id, kIngress, read_args, bytes_count);
    }

    active_read_args_map.delete(&id);
    return 0;
}
"""

'''
C-compatible data structures
'''

char_arr14_t = ct.c_char * 14
in_addr_t = ct.c_uint32
char_arr8_t = ct.c_char * 8


class SockAddr(ct.Structure):
    _fields_ = [
        ('sa_len', ct.c_uint8),
        ('sa_family', ct.c_uint8),
        ('sa_data', char_arr14_t)
    ]

class SockAddr_In(ct.Structure):
    _fields_ = [
        ('sa_len', ct.c_uint8),
        ('sa_family', ct.c_uint8),
        ('sin_port', ct.c_uint16),
        ('sin_addr', in_addr_t),
        ('sin_zero', char_arr8_t)
    ]

class conn_id_t(ct.Structure):
    _fields_ = [("pid", ct.c_ulonglong),
                ("fd", ct.c_ulonglong),
                ("tsid", ct.c_ulonglong)]

class attr_t(ct.Structure):
    _fields_ = [("timestamp_ns", ct.c_ulonglong),
                ("pid", ct.c_uint32),
                ("direction", ct.c_int),
                ("msg_size", ct.c_ulonglong)]

class socket_open_event_t(ct.Structure):
    _fields_ = [("timestamp_ns", ct.c_ulonglong),
                ("conn_id", conn_id_t),
                ("addr", SockAddr_In)]

class socket_data_event_t(ct.Structure):
    _fields_ = [("attr", attr_t),
                ("msg", ct.c_char * MAX_MSG_SIZE)]

class socket_close_event_t(ct.Structure):
    _fields_ = [("timestamp_ns", ct.c_ulonglong),
                ("conn_id", conn_id_t),
                ("wr_bytes", ct.c_ulonglong),
                ("rd_bytes", ct.c_ulonglong)]

examples = """Examples:
    sudo python3 http2_filter           # trace all applications
    sudo python3 http2_filter -p 420    # trace only PID 420
"""

parser = argparse.ArgumentParser(
    description = "Capture and parse HTTP/2 read/write syscall data",
    formatter_class = argparse.RawDescriptionHelpFormatter,
    epilog = examples)

parser.add_argument("-p", "--pid",
        help="Trace only this PID")
args = parser.parse_args()

# if args.pid:
#     bpf_text = bpf_text.replace('FILTER_PID',
#         'if (pid != %s) { return 0; }' % args.pid)

# bpf_text = bpf_text.replace('FILTER_PID', '')

b = BPF(text=bpf_text)

def print_event(cpu, data, size):

    # event = b["socket_open_events"].event(data)
    event = ct.cast(data, ct.POINTER(socket_data_event_t)).contents
    buffer = event.msg
    the_encoding = chardet.detect(buffer)['encoding']
    # if the_encoding != None:
        # print(the_encoding)
        # print(buffer.decode(the_encoding))
        # buffer = ct.string_at(event.msg, event.attr.msg_size)
        # buffer = buffer.from_bytes(ct.string_at(event.msg, event.attr.msg_size), byteorder='big')
        # print("%d %d %s" % (event.attr.timestamp_ns, event.attr.pid, "Output"))
        # print("%d %s %s" % (event.attr.timestamp_ns, buffer.decode(the_encoding), "Output"))
    print("%s", event.msg)
    print("%d %d %s" % (event.attr.timestamp_ns, event.attr.pid, "Output"))
    print(event.attr.pid)

print("Ready")

b["socket_data_events"].open_perf_buffer(print_event)
while 1:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        exit()
