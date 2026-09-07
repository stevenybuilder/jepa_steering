import sys, time, json, numpy as np
sys.path.insert(0, "/root/cgs-pilot/code/cgs_pilot")
import metadrive_hazard_pilot as gen
from metadrive.constants import TerminationState as TS
env = gen.make_env()
for k in ("crash_human_done", "crash_object_done", "crash_vehicle_done", "out_of_road_done"):
    env.config[k] = True
print("horizon", env.config["horizon"], "flags", {k: env.config[k] for k in ("crash_human_done","crash_object_done","crash_vehicle_done","out_of_road_done")})
KEYS = ["arrive_dest","out_of_road","max_step","crash","crash_vehicle","crash_human","crash_object","crash_building","crash_sidewalk"]
def run(seed, hazard, tb, max_steps=1100, label=""):
    map_cfg = gen.scene_map_config(seed, "factorial")
    gen.reset_scene(env, seed, map_cfg)
    env.agent.set_velocity([1.0, 0.0], 8.0, in_local_frame=True)
    obj = None
    if hazard is not None:
        pos, heading = gen.lane_pose(env, 9.0 + hazard["pt"], hazard["lat"]); obj = gen.spawn_hazard_at(env, hazard["kind"], pos, heading, hazard["solid"])
    gen.pose_pedestrian_animation()
    s0 = gen.ego_state(env)
    for t in range(5):
        gen.pose_pedestrian_animation(); env.step([0.0, 0.2])
    pt = float(np.hypot(*(gen.ego_state(env)[:2] - s0[:2])))
    lane = gen.ego_lane(env); l0, _ = lane.local_coordinates(env.agent.position)
    t0 = time.time(); n = 0; done = False; info = {}
    while not done and n < max_steps:
        gen.pose_pedestrian_animation(); obs, r, term, trunc, info = env.step([0.0, tb]); n += 1; done = bool(term or trunc)
    l1, lat1 = lane.local_coordinates(env.agent.position)
    print(f"{label:28s} map {map_cfg} env_steps {n:4d} done={done} term={term} trunc={trunc} wall {time.time()-t0:5.1f}s progress {l1-l0:6.1f} m lat {lat1:+.2f} speed {env.agent.speed:5.2f} route_completion {info.get('route_completion'):.3f} final_lane_len {env.agent.navigation.final_lane.length:.1f} done_info {{{', '.join(k+'='+str(int(bool(info.get(k)))) for k in KEYS)}}}", flush=True)
    if obj is not None: env.engine.clear_objects([obj.id], force_destroy=True)
    return pt
pt = run(9, None, 0.5, label="seed9 free throttle")
run(9, None, 0.0, label="seed9 free coast")
run(9, {"kind":"ped","lat":0.0,"solid":True,"pt":pt}, 0.5, label="seed9 H1 solid throttle")
run(9, {"kind":"ped","lat":0.0,"solid":True,"pt":pt}, -1.0, label="seed9 H1 solid brake->max_step")
run(9, {"kind":"cone","lat":0.0,"solid":False,"pt":pt}, 0.5, label="seed9 H3 ghost throttle")
run(9, {"kind":"ped","lat":2.25,"solid":True,"pt":pt}, 0.5, label="seed9 H0 sidewalk throttle")
pt3 = run(3, None, 0.5, label="seed3 (SCC) free throttle")
for s in (10, 19, 35):
    run(s, None, 0.5, label=f"seed{s} free throttle")
env.close()
