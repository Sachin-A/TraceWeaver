# Algorithms

## Implemented modules

| Algorithm              | Variants                  |
| :----------------------| :-------------------------|
| TraceWeaver            | traceweaver_v*.py         |
| Thread-based approaches| vpath.py, vpath_old.py    |
| Statistical approaches | wap5.py                   |
| First-come-first-serve | fcfs.py, arrival_order.py |

## Interface

### Class template

A custom algorithm must be described via a class with the following template which can be extended to suit the algorithm's needs (refer to fcfs.py for an example).

```
class CustomAlgorithm(object):

    def __init__(self, all_spans, all_processes):
        """
        Initializes the CustomAlgorithm class with given parameters.

        Parameters:
        all_spans (dict): Dictionary where keys are span_ids and the values are the corresponding Span objects.
        all_processes (dict): Nested dictionary where keys are trace_ids and the values are dictionaries mapping process_ids to service_names.
        """

        self.all_spans = all_spans
        self.all_processes = all_processes

    def FindAssignments(self, in_span_partitions, out_span_partitions):
        """
        A method to perform operations which map spans in in_span_partitions to spans in out_span_partitions.

        Parameters:
        in_span_partitions (dict): Dictionary with incoming endpoint(s') names as keys and a list of Span objects as their corresponding values.
        out_span_partitions (dict): Dictionary with outgoing endpoint(s') names as keys and a list of Span objects as their corresponding values.

        Returns:
        dict: A nested dictionary with outgoing endpoint(s') names as keys and the value is another dictionary where the keys are the incoming spans' full IDs (trace_id, span_id) and the values are the corresponding full IDs of the outgoing spans to which each incoming span is mapped to for the given outgoing endpoint (as denoted by the higher level dictionary key).
        """

        # Example processing logic
        result = {}  # Initialize an appropriate data structure for results as described by the return statement.

        '''
        	Custom mapping logic
        '''

        return result
```

### Inputs

Most algorithms take different inputs based off their unique requirements but at the very least, the following must be supplied:

```
1. in_span_partitions: a dictionary with incoming endpoint(s') names as keys and a list of Span objects as their corresponding values. Please check spans.py for the definition of the Span class.
2. out_span_partitions: a dictionary with outgoing endpoint(s') names as keys and a list of Span objects as their corresponding values.
```

### Outputs

All algorithms must return a dictionary mapping incoming spans to outgoing spans per outgoing endpoint.

```
1. result: a nested dictionary with outgoing endpoint(s') names as keys and the value is another dictionary where the keys are the incoming spans' full IDs (trace_id, span_id) and the values are the corresponding full IDs of the outgoing spans to which each incoming span is mapped to for the given outgoing endpoint (as denoted by the higher level dictionary key).

```
