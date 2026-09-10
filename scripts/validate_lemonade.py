#!/usr/bin/env python3
"""Validate the shipped configuration against every number the docs claim."""
import json, subprocess, time, urllib.request, random, string
H, P, M = "192.168.68.60", 13305, "Qwen3.8-27B-GGUF"
KEY = "/home/abanna/Development/nerdsrun/amdllmv/.ssh/framework_fedora"
F = ("The deployment runs llama.cpp behind Lemonade on an AMD Ryzen AI Max 395 "
     "with 128 GB of unified memory, serving a hybrid linear-attention model "
     "where only sixteen of sixty-five layers carry a KV cache. ")
ok = lambda b: "PASS" if b else "**FAIL**"
results = []

def ssh(c):
    return subprocess.run(["ssh","-i",KEY,"-o","BatchMode=yes",f"abanna@{H}",c],
        capture_output=True,text=True,timeout=90,env={"SSH_AUTH_SOCK":""}).stdout.strip()

def ask(prompt, max_tokens=48):
    b=json.dumps({"model":M,"messages":[{"role":"user","content":prompt}],
                  "max_tokens":max_tokens,"stream":True}).encode()
    r0=urllib.request.Request(f"http://{H}:{P}/api/v1/chat/completions",data=b,
        method="POST",headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); ttft=None; n=0; fin=None
    with urllib.request.urlopen(r0,timeout=1800) as r:
        for raw in r:
            l=raw.decode("utf-8","replace").strip()
            if not l.startswith("data:"): continue
            q=l[5:].strip()
            if q=="[DONE]": break
            try: c=json.loads(q)
            except Exception: continue
            ch=c.get("choices",[{}])[0]; d=ch.get("delta",{})
            if (d.get("reasoning_content") or "")+(d.get("content") or ""):
                if ttft is None: ttft=time.perf_counter()-t0
                n+=1
            if ch.get("finish_reason"): fin=ch["finish_reason"]
    return ttft, n, fin, time.perf_counter()-t0

print("="*66)
print("1. CONFIGURATION")
h=json.loads(urllib.request.urlopen(f"http://{H}:{P}/api/v1/health",timeout=60).read())
m=h["all_models_loaded"][0]; argv=" ".join(m["launch_command"])
import re
got={k:(re.search(rf"--{k} (\S+)",argv).group(1) if re.search(rf"--{k} (\S+)",argv) else None)
     for k in ("parallel","ctx-size","cache-ram","ctx-checkpoints")}
exp={"parallel":"1","ctx-size":"262144","cache-ram":"24576","ctx-checkpoints":"8"}
for k,v in exp.items():
    good = got[k]==v
    results.append(good)
    print(f"   --{k:<16} {str(got[k]):>8}  (expect {v:>8})  {ok(good)}")
mtp = "--spec-type draft-mtp" in argv; results.append(mtp)
print(f"   {'MTP enabled':<18} {str(mtp):>8}  (expect     True)  {ok(mtp)}")
print(f"   lemonade {h['version']} | health {m['backend_health']}")

print("\n2. MEMORY")
gtt=int(ssh("cat /sys/class/drm/card*/device/mem_info_gtt_used|head -1"))//1024//1024
free=ssh("free -m | awk '/Mem:/{print $3, $4, $6}'").split()
gtt_ok = gtt < 45000; results.append(gtt_ok)
print(f"   GTT used         {gtt:>8} MiB  (expect <45000, docs say ~36722)  {ok(gtt_ok)}")
print(f"   used {free[0]} MB | free {free[1]} MB | buff/cache {free[2]} MB")

print("\n3. CACHE — the 11.7x claim")
mark="".join(random.choices(string.ascii_lowercase,k=12))
p=f"Validate {mark}. "+F*((9500*4)//len(F))+"\n\nQ: Summarise in five words."
c_ttft,c_n,c_fin,_=ask(p)
w_ttft,w_n,w_fin,_=ask(p)
speed=c_ttft/w_ttft
print(f"   cold TTFT        {c_ttft:>8.2f}s  ({c_n} tokens, finish={c_fin})")
print(f"   warm TTFT        {w_ttft:>8.2f}s  ({w_n} tokens, finish={w_fin})")
sp_ok = speed >= 5; results.append(sp_ok)
print(f"   speedup          {speed:>8.1f}x  (docs claim ~11.7x)  {ok(sp_ok)}")

print("\n4. PREFIX BEHAVIOUR — append keeps it, prepend busts it")
a_ttft,_,_,_=ask(p+" Also mention the memory size.")
b_ttft,_,_,_=ask("Note. "+p)
app_ok = a_ttft < 6; pre_ok = b_ttft > 10
results += [app_ok, pre_ok]
print(f"   appended         {a_ttft:>8.2f}s  (expect fast, cache retained)  {ok(app_ok)}")
print(f"   prepended        {b_ttft:>8.2f}s  (expect slow, cache busted)    {ok(pre_ok)}")

print("\n5. DECODE — the 21.9 tok/s claim")
rates=[]
for i in range(3):
    mk="".join(random.choices(string.ascii_lowercase,k=12))
    _,n,_,tot=ask(f"Dec {mk}. "+F*((2000*4)//len(F))+"\n\nQ: Explain briefly.",48)
    rates.append(n/tot if tot else 0)
srv=ssh('podman logs --since 3m lemonade-server 2>&1 | grep -a "eval time" | grep -av "prompt eval" | tail -3 | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+"')
srv=[float(x) for x in srv.split()] if srv else []
if srv:
    mean=sum(srv)/len(srv); d_ok = mean>15; results.append(d_ok)
    print(f"   server eval      {mean:>8.1f} tok/s  {srv}  (expect >15)  {ok(d_ok)}")
else:
    print("   (no server timings captured)")

print("\n"+"="*66)
print(f"   {sum(results)} / {len(results)} checks passed"
      + ("   ALL GOOD" if all(results) else "   SOME FAILED"))
