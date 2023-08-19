from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent.parent

def GetOutEpsInOrder(out_span_partitions):
    eps = []
    for ep, spans in out_span_partitions.items():
        assert len(spans) > 0
        eps.append((ep, spans[0].start_mus))
    eps.sort(key=lambda x: x[1])
    return [x[0] for x in eps]
