#include "span.h"

Span::Span(
    string span_id, string trace_id, string parent_span_id,
    string caller, string callee, int log_location,
    long int request_start_mus, long int response_end_mus,
    vector<string> metadata, long int request_end_mus = -1,
    long int response_start_mus = -1)
    : span_id_(span_id), trace_id_(trace_id), parent_span_id_(parent_span_id),
      caller_(caller), callee_(callee), log_location_(log_location),
      request_start_mus_(request_start_mus), response_end_mus_(response_end_mus),
      metadata_(metadata), request_end_mus_(request_end_mus),
      response_start_mus_(response_start_mus)
{
}

long int Span::getDuration()
{
    return response_end_mus_ - request_start_mus_;
}
