#ifndef __DISTTRACE_SPAN__
#define __DISTTRACE_SPAN__

#include <iostream>
#include <string>
#include <vector>
using namespace std;

typedef string TraceId;
typedef string SpanId; 

class Span
{
    TraceId trace_id;
    SpanId span_id;
    SpanId parent_span_id;
    vector<SpanId> child_span_ids;
    string caller, callee;
    int log_location;
    long int request_start_mus, request_end_mus;
    long int response_start_mus, response_end_mus;
    vector<string> metadata;

public:
    Span(SpanId span_id_, string trace_id_, string parent_span_id_,
         string caller_, string callee_, int log_location_,
         long int request_start_mus_, long int response_end_mus_,
         vector<string> metadata_, long int request_end_mus_,
         long int response_start_mus_);

    long int GetDuration();
    void AddChild(SpanId child);
    pair<TraceId, SpanId> GetId();
};

#endif
