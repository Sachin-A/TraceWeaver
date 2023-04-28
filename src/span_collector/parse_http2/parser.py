import re
import h2.connection
import h2.config
import codecs
import logging

from enum import Enum

class Direction(Enum):
    INGRESS = 1
    EGRESS = 2

# (uber-trace-id, uber-span-id): [thread-id, uber-parent-id, x-request-id, x-b3-traceid, x-b3-spanid]
request_to_thread_map = {}

# logging.TRACE = logging.INFO + 5
# logging.addLevelName(logging.INFO + 5, 'TRACE')
# class MyLogger(logging.getLoggerClass()):
#     def trace(self, msg, *args, **kwargs):
#         self.log(logging.TRACE, msg, *args, **kwargs)
# logging.setLoggerClass(MyLogger)
# logging.basicConfig(level=logging.NOTSET)
# logger = logging.getLogger('http2_logger')

# def send_response(conn, event):
#     stream_id = event.stream_id
#     response_data = json.dumps(dict(event.headers)).encode('utf-8')

#     conn.send_headers(
#         stream_id=stream_id,
#         headers=[
#             (':status', '200'),
#             ('server', 'basic-h2-server/1.0'),
#             ('content-length', str(len(response_data))),
#             ('content-type', 'application/json'),
#         ],
#     )
#     conn.send_data(
#         stream_id=stream_id,
#         data=response_data,
#         end_stream=True
#     )

def map_request_to_thread(event, entry, stream_to_request_map, fd):
    request_id = None

    for header in event.headers:
        if header[0] == b'uber-trace-id':
            values = header[1].decode('utf-8').split(':')
            request_id = (values[0], values[1])
            request_to_thread_map[request_id] = [None] * 7
            request_to_thread_map[request_id][0] = int(entry[0])
            request_to_thread_map[request_id][1] = values[2]
            request_to_thread_map[request_id][6] = fd

    for header in event.headers:
        if request_id is not None:
            if header[0] == b'x-request-id':
                value = header[1].decode('utf-8')
                request_to_thread_map[request_id][2] = value
            if header[0] == b'x-b3-traceid':
                value = header[1].decode('utf-8')
                request_to_thread_map[request_id][3] = value
            if header[0] == b'x-b3-spanid':
                value = header[1].decode('utf-8')
                request_to_thread_map[request_id][4] = value

    stream_to_request_map[event.stream_id] = request_id

def map_response_to_thread(event, entry, stream_to_request_map):
    if event.stream_id not in stream_to_request_map:
        assert False
    if stream_to_request_map[event.stream_id] not in request_to_thread_map:
        assert False
    request_to_thread_map[stream_to_request_map[event.stream_id]][5] = int(entry[0])

def handle3(data, dir, bidirectional_data, fd):

    am_i_client = True
    if dir == Direction.INGRESS:
        am_i_client = False

    # config = h2.config.H2Configuration(client_side=am_i_client, logger=logger)
    config = h2.config.H2Configuration(client_side=am_i_client)
    conn = h2.connection.H2Connection(config=config)

    # config_peer = h2.config.H2Configuration(client_side=not am_i_client, logger=logger)
    config_peer = h2.config.H2Configuration(client_side=not am_i_client)
    conn_peer = h2.connection.H2Connection(config=config_peer)

    count = 0
    error_count = 0

    event_list = set()
    stream_to_request_map = {}

    for entry in bidirectional_data:

        bytestream = entry[1]

        count += 1
        print("Entry num: ", count, "Type: ", entry[2])

        if entry[2] == Direction.EGRESS:

            # print(bytestream)

            try:
                # print("Server sees")
                events = conn_peer.receive_data(bytestream)

                for event in events:
                    # print(event)
                    # input()
                    event_list.add(event.__class__)

                    if isinstance(event, h2.events.RequestReceived):
                        map_request_to_thread(event, entry, stream_to_request_map, fd)

                        # Update client state machine
                        conn.send_headers(stream_id=event.stream_id, headers=event.headers, end_stream=False, priority_weight=None, priority_depends_on=None, priority_exclusive=None)
                        conn.send_data(stream_id=event.stream_id, data=b'it works!', end_stream=True)
                        # conn._begin_new_stream(1, allowed_ids=1)

                    if isinstance(event, h2.events.ResponseReceived):
                        map_response_to_thread(event, entry, stream_to_request_map)

                    if isinstance(event, h2.events.StreamEnded):
                        # Update server state machine
                        conn_peer.end_stream(int(event.stream_id))

            except Exception as e:
                print("Protocol error (server-side): ", e)
                error_count += 1
                input()

        elif entry[2] == Direction.INGRESS:

            try:
                # print("Client sees")
                events = conn.receive_data(bytestream)

                for event in events:
                    event_list.add(event.__class__)

                    if isinstance(event, h2.events.RequestReceived):
                        map_request_to_thread(event, entry, stream_to_request_map, fd)

                    if isinstance(event, h2.events.ResponseReceived):
                        map_response_to_thread(event, entry, stream_to_request_map)

            except Exception as e:
                print("Protocol error (client-side): ", e)
                error_count += 1
                input()

    print("Overall count: ", count)
    print("Overall error count: ", error_count)
    print("Overall event list: ", event_list)

def handle2(data, dir, bidirectional_data, fd):

    am_i_client = True
    if dir == Direction.INGRESS:
        am_i_client = False

    # config = h2.config.H2Configuration(client_side=am_i_client, logger=logger)
    config = h2.config.H2Configuration(client_side=am_i_client)
    conn = h2.connection.H2Connection(config=config)

    # config_peer = h2.config.H2Configuration(client_side=not am_i_client, logger=logger)
    config_peer = h2.config.H2Configuration(client_side=not am_i_client)
    conn_peer = h2.connection.H2Connection(config=config_peer)

    count = 0
    error_count = 0

    event_list = set()
    stream_to_request_map = {}

    for entry in bidirectional_data:

        bytestream = entry[1]

        count += 1
        print("Entry num: ", count, "Type: ", entry[2])
        print(count)

        if entry[2] == Direction.INGRESS:

            try:
                # print("Server sees")
                events = conn_peer.receive_data(bytestream)

                for event in events:
                    # print(event)
                    # input()
                    event_list.add(event.__class__)

                    if isinstance(event, h2.events.RequestReceived):
                        map_request_to_thread(event, entry, stream_to_request_map, fd)

                        # Update client state machine
                        conn.send_headers(stream_id=event.stream_id, headers=event.headers, end_stream=False, priority_weight=None, priority_depends_on=None, priority_exclusive=None)
                        conn.send_data(stream_id=event.stream_id, data=b'it works!', end_stream=True)
                        # conn._begin_new_stream(1, allowed_ids=1)

                    if isinstance(event, h2.events.ResponseReceived):
                        map_response_to_thread(event, entry, stream_to_request_map)

                    if isinstance(event, h2.events.StreamEnded):
                        # Update server state machine
                        conn_peer.end_stream(int(event.stream_id))

            except Exception as e:
                print("Protocol error (server-side): ", e)
                error_count += 1
                input()

        elif entry[2] == Direction.EGRESS:

            try:
                # print("Client sees")
                events = conn.receive_data(bytestream)

                for event in events:
                    event_list.add(event.__class__)

                    if isinstance(event, h2.events.RequestReceived):
                        map_request_to_thread(event, entry, stream_to_request_map, fd)

                    if isinstance(event, h2.events.ResponseReceived):
                        map_response_to_thread(event, entry, stream_to_request_map)

            except Exception as e:
                print("Protocol error (client-side): ", e)
                error_count += 1
                input()

    print(count)
    print(error_count)
    print(event_list)


def handle(data, dir, bidirectional_data):

    count = 0
    error_count = 0

    am_i_client = True
    if dir == Direction.INGRESS:
        am_i_client = False

    # config = h2.config.H2Configuration(client_side=am_i_client, logger=logger)
    config = h2.config.H2Configuration(client_side=am_i_client)
    conn = h2.connection.H2Connection(config=config)

    if am_i_client:
        conn.initiate_connection()
        print(conn.data_to_send())

    event_list = set()

    for entry in data:

        count += 1
        print(count)

        bytestream = entry[1]

        try:
            # if count == 4:
            #     conn._begin_new_stream(1, allowed_ids=1)
            events = conn.receive_data(bytestream)
            for event in events:
                if(dir == Direction.EGRESS or dir == Direction.INGRESS):
                    event_list.add(event.__class__)
                    print(event)
                    print(event.__class__)
                    input()

                if isinstance(event, h2.events.RequestReceived):
                    map_request_to_thread(event, entry)

                if isinstance(event, h2.events.StreamEnded):
                    conn.end_stream(int(event.stream_id))

        except Exception as e:
            print("Protocol error: ", e)
            error_count += 1
            input()

    print(event_list)
    print(count)
    print(error_count)

filename = "./perf_traces/output-node2-exp2-attempt4.log"

pattern1 = r'(\d+) read\((\d+), \"(.*)\", (\d+)\)[ ]{1,}=[ ]{1,}(\d+)'
pattern2 = r'(\d+) read\((\d+),[ ]{1,}<unfinished \.{1,}>'
pattern3 = r'(\d+) <\.{1,}[ ]{1,}read resumed>\"(.*)\", (\d+)\)[ ]{1,}=[ ]{1,}(\d+)'
pattern4 = r'(\d+) write\((\d+), \"(.*)\", (\d+)\)[ ]{1,}=[ ]{1,}(\d+)'
pattern5 = r'(\d+) write\((\d+), \"(.*)\", (\d+)[ ]{1,}<unfinished[ ]{1,}\.{1,}>'
pattern6 = r'(\d+) <\.{1,}[ ]{1,}write resumed>[ ]{0,}\)[ ]{1,}=[ ]{1,}(\d+)'
pattern7 = r'(\d+) close\((\d+)\)[ ]*=[ ]*(\d+)'
pattern8 = r'(\d+) close\((\d+)[ ]*<unfinished[ ]*\.*>'
pattern9 = r'(\d+) <\.*[ ]*close resumed>[ ]*\)[ ]*=[ ]*(\d+)'

per_fd = {}
bytestreams = []
per_fd_bidirectional = {}
read_outstanding_fd = {}
write_outstanding_bytes = {}
write_outstanding_fd = {}
close_outstanding_fd = {}
current_fd_iteration = {}

count = 0

with open(filename) as file:
    while line := file.readline():

        count += 1
        if count % 100 == 0:
            print("Line:", count)

        line = line.rstrip()
        match1 = re.search(pattern1, line)
        match2 = re.search(pattern2, line)
        match3 = re.search(pattern3, line)
        match4 = re.search(pattern4, line)
        match5 = re.search(pattern5, line)
        match6 = re.search(pattern6, line)
        match7 = re.search(pattern7, line)
        match8 = re.search(pattern8, line)
        match9 = re.search(pattern9, line)

        if match1 != None:
            thread_id = match1.group(1)
            fd = match1.group(2)
            bytestream = match1.group(3)
            return_code = int(match1.group(5))

            if fd not in current_fd_iteration:
                current_fd_iteration[fd] = 1

            if return_code <= 0:
                continue

            if (fd, Direction.INGRESS, current_fd_iteration[fd]) not in per_fd:
                per_fd[(fd, Direction.INGRESS, current_fd_iteration[fd])] = []

            if (fd, current_fd_iteration[fd]) not in per_fd_bidirectional:
                per_fd_bidirectional[(fd, current_fd_iteration[fd])] = []

            # per_fd[fd].append((thread_id, bytestream.encode('utf-8').decode('unicode_escape').encode('utf-8')))
            per_fd[(fd, Direction.INGRESS, current_fd_iteration[fd])].append((thread_id, codecs.escape_decode(bytestream, 'utf-8')[0]))
            per_fd_bidirectional[(fd, current_fd_iteration[fd])].append((thread_id, codecs.escape_decode(bytestream, 'utf-8')[0], Direction.INGRESS))
            # bytestreams.append((thread_id, codecs.escape_decode(bytestream, 'utf-8')[0], Direction.INGRESS))

        elif match2 != None:
            thread_id = match2.group(1)
            fd = match2.group(2)

            if thread_id in read_outstanding_fd:
                read_outstanding_fd[thread_id] = (fd, Direction.INGRESS)

            else:
                read_outstanding_fd[thread_id] = (fd, Direction.INGRESS)

        elif match3 != None:
            thread_id = match3.group(1)
            bytestream = match3.group(2)
            return_code = int(match3.group(3))

            fd = read_outstanding_fd[thread_id][0]
            del read_outstanding_fd[thread_id]

            if fd not in current_fd_iteration:
                current_fd_iteration[fd] = 1

            if return_code <= 0:
                continue

            if (fd, Direction.INGRESS, current_fd_iteration[fd]) not in per_fd:
                per_fd[(fd, Direction.INGRESS, current_fd_iteration[fd])] = []

            if (fd, current_fd_iteration[fd]) not in per_fd_bidirectional:
                per_fd_bidirectional[(fd, current_fd_iteration[fd])] = []

            per_fd[(fd, Direction.INGRESS, current_fd_iteration[fd])].append((thread_id, codecs.escape_decode(bytestream, 'utf-8')[0]))
            per_fd_bidirectional[(fd, current_fd_iteration[fd])].append((thread_id, codecs.escape_decode(bytestream, 'utf-8')[0], Direction.INGRESS))
            #bytestreams.append((thread_id, codecs.escape_decode(bytestream, 'utf-8')[0], Direction.INGRESS))

        elif match4 != None:
            thread_id = match4.group(1)
            fd = match4.group(2)
            bytestream = match4.group(3)
            return_code = int(match4.group(5))

            if fd not in current_fd_iteration:
                current_fd_iteration[fd] = 1

            if return_code <= 0:
                continue

            if (fd, Direction.EGRESS, current_fd_iteration[fd]) not in per_fd:
                per_fd[(fd, Direction.EGRESS, current_fd_iteration[fd])] = []

            if (fd, current_fd_iteration[fd]) not in per_fd_bidirectional:
                per_fd_bidirectional[(fd, current_fd_iteration[fd])] = []

            # per_fd[(fd, Direction.EGRESS, current_fd_iteration[fd])].append((thread_id, bytestream.encode('utf-8').decode('unicode_escape').encode('utf-8')))
            per_fd[(fd, Direction.EGRESS, current_fd_iteration[fd])].append((thread_id, codecs.escape_decode(bytestream, 'utf-8')[0]))
            per_fd_bidirectional[(fd, current_fd_iteration[fd])].append((thread_id, codecs.escape_decode(bytestream, 'utf-8')[0], Direction.EGRESS))
            # bytestreams.append((thread_id, codecs.escape_decode(bytestream, 'utf-8')[0], Direction.EGRESS))

        elif match5 != None:
            thread_id = match5.group(1)
            fd = match5.group(2)
            bytestream = match5.group(3)

            write_outstanding_fd[thread_id] = fd

            if (thread_id, fd) not in write_outstanding_bytes:
                write_outstanding_bytes[(thread_id, fd)] = []

            write_outstanding_bytes[(thread_id, fd)].append(bytestream)

        elif match6 != None:
            thread_id = match6.group(1)
            return_code = int(match6.group(2))
            print(count)
            print(write_outstanding_fd.keys())
            # input()
            fd = write_outstanding_fd[thread_id]
            del write_outstanding_fd[thread_id]

            if fd not in current_fd_iteration:
                current_fd_iteration[fd] = 1

            if return_code <= 0:
                continue

            if (fd, Direction.EGRESS, current_fd_iteration[fd]) not in per_fd:
                per_fd[(fd, Direction.EGRESS, current_fd_iteration[fd])] = []

            if (fd, current_fd_iteration[fd]) not in per_fd_bidirectional:
                per_fd_bidirectional[(fd, current_fd_iteration[fd])] = []

            combined_write_data = ''.join(write_outstanding_bytes[(thread_id, fd)])
            del write_outstanding_bytes[(thread_id, fd)]
            per_fd[(fd, Direction.EGRESS, current_fd_iteration[fd])].append((thread_id, codecs.escape_decode(combined_write_data, 'utf-8')[0]))
            per_fd_bidirectional[(fd, current_fd_iteration[fd])].append((thread_id, codecs.escape_decode(combined_write_data, 'utf-8')[0], Direction.EGRESS))
            # bytestreams.append((thread_id, codecs.escape_decode(bytestream, 'utf-8')[0], Direction.EGRESS))

        elif match7 != None:
            thread_id = match7.group(1)
            fd = match7.group(2)
            return_code = int(match7.group(3))

            if return_code == 0:
                if fd not in current_fd_iteration:
                    current_fd_iteration[fd] = 1
                current_fd_iteration[fd] += 1

        elif match8 != None:
            thread_id = match8.group(1)
            fd = match8.group(2)

            close_outstanding_fd[thread_id] = fd

        elif match9 != None:
            thread_id = match9.group(1)
            return_code = int(match9.group(2))
            fd = close_outstanding_fd[thread_id]
            del close_outstanding_fd[thread_id]

            if return_code < 0:
                continue

            elif return_code == 0:
                if fd not in current_fd_iteration:
                    current_fd_iteration[fd] = 1
                current_fd_iteration[fd] += 1

for (fd, iteration) in per_fd_bidirectional.keys():
    print(fd, iteration, len(per_fd_bidirectional[fd]))

input()

for (fd, iteration) in per_fd_bidirectional.keys():
    if iteration == current_fd_iteration[fd]:
        if fd in ["11", "12", "13"]:
            print("FD: ", fd)
            # input()
            if fd == "13":
                handle2(per_fd_bidirectional[fd, current_fd_iteration[fd]], Direction.EGRESS, per_fd_bidirectional[fd, current_fd_iteration[fd]], fd)
            elif fd == "11" or fd == "12":
                handle3(per_fd_bidirectional[fd, current_fd_iteration[fd]], Direction.EGRESS, per_fd_bidirectional[fd, current_fd_iteration[fd]], fd)
            # elif fd == "14":
                # handle2(per_fd_bidirectional[fd], Direction.EGRESS, per_fd_bidirectional[fd])

# handle3(bytestreams, Direction.EGRESS, bytestreams)

none_count = 0
for key, values in request_to_thread_map.items():
    # print(values)
    # input()
    indices = [0, 5]
    index = 0
    for value in values:
        if index in indices and value == None:
            none_count += 1
        index += 1

print("num_keys: ", len(request_to_thread_map.keys()))
print("none_count: ", none_count)

per_trace_id = {}
fd_to_index = {
    "11": 1,
    "12": 2,
    "13": 0
}

unique_threads = set()

for k1, v1 in request_to_thread_map.items():
    if k1[0] not in per_trace_id:
        per_trace_id[k1[0]] = [None] * 3
    per_trace_id[k1[0]][fd_to_index[v1[-1]]] = [v1[0], v1[-2]]
    unique_threads.add(v1[0])
    unique_threads.add(v1[-2])

for k, v in per_trace_id.items():
    print(k)
    print(v)
    input()

unique_threads = list(unique_threads)

X = []
y = []

one_hot = [0] * len(unique_threads)

count = 0
for k1 in per_trace_id:
    count += 1
    a = list(one_hot)
    b = list(one_hot)
    c = list(one_hot)
    d = list(one_hot)
    e = list(one_hot)
    a[unique_threads.index(per_trace_id[k1][0][0])] = 1
    b[unique_threads.index(per_trace_id[k1][0][1])] = 1
    c[unique_threads.index(per_trace_id[k1][1][0])] = 1
    d[unique_threads.index(per_trace_id[k1][1][1])] = 1
    features = sum([a, b, c, d], [])

    e = unique_threads.index(per_trace_id[k1][2][0])
    # X.append([per_trace_id[k1][0][0], per_trace_id[k1][0][1], per_trace_id[k1][1][0], per_trace_id[k1][1][1]])
    # y.append(per_trace_id[k1][2][0])
    X.append(features)
    y.append(e)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
