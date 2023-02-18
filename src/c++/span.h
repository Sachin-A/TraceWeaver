#ifndef __DISTTRACE_SPAN__
#define __DISTTRACE_SPAN__

#include <iostream>
#include <string>
#include <vector>
using namespace std;

// pair <trace_id, span_id>
typedef pair<string, string> Reference; 

class Span{
    string span_id;
    string trace_id;
    string caller, callee;
    long int start_mus, duration_mus;
    vector<Reference> references;
    
public:
    Span();
};

#endif
