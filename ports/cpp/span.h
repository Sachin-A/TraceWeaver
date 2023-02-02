#ifndef __DISTTRACE_SPAN__
#define __DISTTRACE_SPAN__

#include <iostream>
#include <string>
#include <vector>
using namespace std;

class Span
{
    string span_id_, trace_id_;
    string parent_span_id_;
    string caller_, callee_;
    int log_location_;
    long int request_start_mus_, request_end_mus_;
    long int response_start_mus_, response_end_mus_;
    vector<string> metadata_;

public:
    Span(string span_id, string trace_id, string parent_span_id, string caller,
         string callee, int log_location, long int request_start_mus,
         long int response_end_mus, vector<string> metadata,
         long int request_end_mus, long int response_start_mus);

    long int getDuration();
};

#endif
