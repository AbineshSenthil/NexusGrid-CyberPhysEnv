import traceback
from server.dashboard import on_reset, on_step, on_auto_run, create_power_flow_sankey, create_topology_graph

try:
    r = on_reset(0, 42)
    print("reset", len(r))
    r2 = on_step("dispatch_generation", "NODE_01", "LINE_01", 100, "CLOSED", -0.5, 5, '["NODE_14"]')
    print("step", len(r2))
    r3 = on_auto_run(0, 42, 3)
    print("auto", len(r3))
except Exception:
    traceback.print_exc()
