#include "span.h"

Span::Span(
    string span_id_, string trace_id_, string parent_span_id_,
    string caller_, string callee_, int log_location_,
    long int request_start_mus_, long int response_end_mus_,
    vector<string> metadata_, long int request_end_mus_ = -1,
    long int response_start_mus_ = -1)
    : span_id(span_id_), trace_id(trace_id_), parent_span_id(parent_span_id_),
      caller(caller_), callee(callee_), log_location(log_location_),
      request_start_mus(request_start_mus_), response_end_mus(response_end_mus_),
      metadata(metadata_), request_end_mus(request_end_mus_),
      response_start_mus(response_start_mus_)
{
}

long int Span::GetDuration()
{
    return response_end_mus - request_start_mus;
}
