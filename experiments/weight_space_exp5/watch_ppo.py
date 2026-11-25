"""Read-only local browser viewer for a separately played PPO Space Invaders game."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch
from minatar import Environment

from exp5 import CHANNELS, observation_indices, policy_action


PAGE = r"""<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MinAtar Space Invaders · PPO 관전</title>
<style>
  :root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#091421;color:#e8f2ff}
  body{max-width:960px;margin:0 auto;padding:22px} h1{font-size:23px;margin:0 0 8px}
  .note{color:#b6c8d8;line-height:1.5;margin:0 0 16px}
  .layout{display:flex;gap:24px;flex-wrap:wrap}.card{background:#122337;border:1px solid #274157;border-radius:16px;padding:18px}
  canvas{display:block;width:min(82vw,500px);height:min(82vw,500px);image-rendering:pixelated;background:#0d2033}
  .side{flex:1;min-width:230px}.label{color:#a8bfd4;font-size:13px}.value{font-size:22px;font-weight:700;margin:2px 0 15px}
  button{background:#24435c;color:white;border:1px solid #57738a;border-radius:8px;padding:9px 13px;margin:0 7px 10px 0;cursor:pointer}
  button.active{background:#3574a4}.bar{height:11px;background:#253c51;border-radius:8px;overflow:hidden;margin:7px 0 13px}
  .fill{height:100%;background:#55c4ec;width:0}.legend{display:flex;flex-wrap:wrap;gap:10px;margin-top:15px;color:#b6c8d8;font-size:13px}
  .swatch{display:inline-block;width:10px;height:10px;margin-right:4px}.small{font-size:13px;color:#a8bfd4}
</style>
<h1>MinAtar Space Invaders · PPO 정책 관전</h1>
<p class="note">최신 중간 저장 가중치로 <b>별도 게임</b>을 플레이합니다. 실제 학습 중인 64개 게임의 화면은 아닙니다. 정책은 중간 저장 시점마다 갱신됩니다.</p>
<p class="note"><a style="color:#75d8ff" href="/training">실제 학습 중인 64개 게임 보기 →</a></p>
<div class="layout"><div class="card"><canvas id="board" width="500" height="500"></canvas>
<div class="legend"><span><i class="swatch" style="background:#62b5ff"></i>플레이어</span><span><i class="swatch" style="background:#78e057"></i>적</span><span><i class="swatch" style="background:#ffd54a"></i>내 탄환</span><span><i class="swatch" style="background:#ff5e67"></i>적 탄환</span></div></div>
<div class="card side"><div><button id="seed1" onclick="selectSeed(1)">시드 1</button><button id="seed2" onclick="selectSeed(2)">시드 2</button><button id="pause" onclick="togglePause()">일시정지</button></div>
<div class="label">시범 플레이 점수</div><div class="value" id="score">—</div>
<div class="label">시범 플레이 회차 · 스텝</div><div class="value" id="episode">—</div>
<div class="label">현재 사용 중인 정책 저장 시점</div><div class="value" id="checkpoint">—</div>
<div class="label">학습 진행 (최대 5억 스텝)</div><div class="value" id="progress">—</div><div class="bar"><div class="fill" id="fill"></div></div>
<div class="small" id="status">연결 중…</div><p class="small">속도 <input id="speed" type="range" min="2" max="20" value="8"> <span id="speedValue">8</span> 스텝/초</p>
</div></div>
<script>
let seed=1, paused=false, busy=false;
const board=document.getElementById('board'), c=board.getContext('2d');
function selectSeed(n){seed=n;document.getElementById('seed1').classList.toggle('active',n===1);document.getElementById('seed2').classList.toggle('active',n===2)}
function togglePause(){paused=!paused;document.getElementById('pause').textContent=paused?'재생':'일시정지'}
selectSeed(1);
function draw(ch){
  c.fillStyle='#0d2033';c.fillRect(0,0,500,500);
  for(let y=0;y<10;y++)for(let x=0;x<10;x++){
    let t=ch[y][x],X=x*50,Y=y*50;
    c.fillStyle='#132b40';c.fillRect(X+1,Y+1,48,48);
    if(t[1]){c.fillStyle='#78e057';c.fillRect(X+7,Y+10,36,30)}
    if(t[4]){c.fillStyle='#ffd54a';c.fillRect(X+22,Y+7,7,36)}
    if(t[5]){c.fillStyle='#ff5e67';c.beginPath();c.arc(X+25,Y+25,9,0,Math.PI*2);c.fill()}
    if(t[0]){c.fillStyle='#62b5ff';c.beginPath();c.moveTo(X+25,Y+7);c.lineTo(X+6,Y+43);c.lineTo(X+44,Y+43);c.fill()}
  }
}
async function tick(){
  if(!paused&&!busy){busy=true;try{
    const r=await fetch('/api/frame?seed='+seed,{cache:'no-store'});const d=await r.json();
    if(!r.ok)throw new Error(d.error||r.statusText);
    draw(d.channels);document.getElementById('score').textContent=d.score+(d.done?' · 게임 종료':'');
    document.getElementById('episode').textContent=d.episode+'회차 · '+d.game_step+'스텝';
    document.getElementById('checkpoint').textContent=(d.checkpoint_steps/1e6).toFixed(2)+'M 스텝';
    document.getElementById('progress').textContent=(d.training_steps/1e6).toFixed(2)+'M / 500M ('+(d.training_steps/5e6).toFixed(1)+'%)';
    document.getElementById('fill').style.width=Math.min(100,d.training_steps/5e6)+'%';
    document.getElementById('status').textContent='시드 '+seed+' · 마지막 행동 '+d.action_name+' · '+(d.completed?'학습 완료':'학습 진행 중');
  }catch(e){document.getElementById('status').textContent='연결 오류: '+e.message}finally{busy=false}}
  const speed=Number(document.getElementById('speed').value);document.getElementById('speedValue').textContent=speed;
  setTimeout(tick,1000/speed);
}
tick();
</script></html>"""


TRAINING_PAGE = r"""<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MinAtar · 실제 학습 게임 64개</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#091421;color:#e8f2ff}
body{max-width:1200px;margin:0 auto;padding:20px}h1{font-size:23px;margin:0 0 8px}
p{color:#b6c8d8;line-height:1.5}a{color:#75d8ff}
button{background:#24435c;color:#fff;border:1px solid #57738a;border-radius:8px;padding:9px 13px;cursor:pointer;margin-right:8px}
button.active{background:#3574a4}.toolbar{margin:15px 0}.status{font-size:14px;color:#a8bfd4;margin:12px 0}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:20px}
.grid{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:6px}
.tile{background:#122337;border:1px solid #274157;border-radius:6px;padding:3px;cursor:pointer;text-align:center}
.tile.selected{border:2px solid #75d8ff;padding:2px}.tile canvas{width:100%;aspect-ratio:1;image-rendering:pixelated;display:block}
.tile span{font-size:11px;color:#a8bfd4}.detail{background:#122337;border:1px solid #274157;border-radius:12px;padding:14px;align-self:start}
#large{width:100%;aspect-ratio:1;image-rendering:pixelated;display:block;background:#0d2033}
.legend{font-size:12px;color:#a8bfd4;line-height:1.8}.swatch{display:inline-block;width:10px;height:10px;margin:0 3px 0 10px}
@media(max-width:800px){.layout{grid-template-columns:1fr}.grid{gap:3px}.tile{padding:1px}.tile span{font-size:9px}}
</style>
<h1>실제 PPO 학습 게임 64개</h1>
<p>학습 프로세스의 64개 환경 상태를 주기적으로 복사한 화면입니다. 각 칸은 서로 독립된 게임이며, 학습 속도를 위해 모든 프레임을 저장하지는 않습니다.</p>
<div class="toolbar"><button id="seed1" onclick="selectSeed(1)">시드 1</button><button id="seed2" onclick="selectSeed(2)">시드 2</button><a href="/">별도 정책 시범 플레이 ↗</a></div>
<div class="status" id="status">연결 중…</div>
<div class="layout"><div class="grid" id="grid"></div><div class="detail"><canvas id="large" width="300" height="300"></canvas>
<h3 id="selectedTitle">게임 1</h3><div id="selectedInfo" class="status">—</div>
<div class="legend"><i class="swatch" style="background:#62b5ff"></i>플레이어<i class="swatch" style="background:#78e057"></i>적<i class="swatch" style="background:#ffd54a"></i>내 탄환<i class="swatch" style="background:#ff5e67"></i>적 탄환</div></div></div>
<script>
let seed=1,selected=0,busy=false,last=null;
const grid=document.getElementById('grid'),tiles=[];
for(let i=0;i<64;i++){const tile=document.createElement('div');tile.className='tile';tile.onclick=()=>{selected=i;render(last)};
 const canvas=document.createElement('canvas');canvas.width=100;canvas.height=100;tile.append(canvas);
 const label=document.createElement('span');label.textContent='게임 '+(i+1);tile.append(label);grid.append(tile);tiles.push({tile,canvas})}
function selectSeed(n){seed=n;last=null;document.getElementById('seed1').classList.toggle('active',n===1);document.getElementById('seed2').classList.toggle('active',n===2)}
selectSeed(1);
function draw(canvas,encoded){const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));const c=canvas.getContext('2d');const scale=canvas.width/10;
 c.fillStyle='#0d2033';c.fillRect(0,0,canvas.width,canvas.height);
 for(let y=0;y<10;y++)for(let x=0;x<10;x++){const t=bytes[y*10+x],X=x*scale,Y=y*scale;
 c.fillStyle='#132b40';c.fillRect(X+1,Y+1,scale-2,scale-2);
 if(t&14){c.fillStyle='#78e057';c.fillRect(X+scale*.15,Y+scale*.2,scale*.7,scale*.6)}
 if(t&16){c.fillStyle='#ffd54a';c.fillRect(X+scale*.45,Y+scale*.1,scale*.12,scale*.8)}
 if(t&32){c.fillStyle='#ff5e67';c.beginPath();c.arc(X+scale*.5,Y+scale*.5,scale*.18,0,Math.PI*2);c.fill()}
 if(t&1){c.fillStyle='#62b5ff';c.beginPath();c.moveTo(X+scale*.5,Y+scale*.12);c.lineTo(X+scale*.12,Y+scale*.85);c.lineTo(X+scale*.88,Y+scale*.85);c.fill()}
 }}
const names=['대기','왼쪽','위쪽','오른쪽','아래쪽','발사'];
function render(d){if(!d)return;last=d;
 for(let i=0;i<64;i++){tiles[i].tile.classList.toggle('selected',i===selected);draw(tiles[i].canvas,d.states[i])}
 draw(document.getElementById('large'),d.states[selected]);
 document.getElementById('selectedTitle').textContent='게임 '+(selected+1);
 document.getElementById('selectedInfo').textContent='현재 에피소드 '+d.episode_steps[selected]+'스텝 · 마지막 행동 '+names[d.actions[selected]]+' · 보상 '+d.rewards[selected]+(d.done[selected]?' · 직전 게임 종료':'');
 const age=Math.max(0,(Date.now()/1000-d.captured_at)).toFixed(1);
 document.getElementById('status').textContent='시드 '+seed+' · '+(d.steps/1e6).toFixed(2)+'M / 500M 스텝 · 갱신 '+age+'초 전'+(age>10?' (학습이 중단되었거나 완료되었을 수 있음)':'');
}
async function tick(){if(!busy){busy=true;try{const r=await fetch('/api/training?seed='+seed,{cache:'no-store'});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);if(d.seed===seed)render(d)}
 catch(e){document.getElementById('status').textContent='화면 대기 중: '+e.message}finally{busy=false}}setTimeout(tick,250)}
tick();
</script></html>"""


ACTION_NAMES = ("대기", "왼쪽", "위쪽", "오른쪽", "아래쪽", "발사")


def recent_steps(log_path, fallback):
    if not log_path.exists():
        return fallback
    with log_path.open("rb") as source:
        source.seek(0, 2)
        size = source.tell()
        source.seek(max(0, size - 4096))
        lines = source.read().decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "steps" in value:
            return max(fallback, int(value["steps"]))
    return fallback


class Spectator:
    def __init__(self, seed, run_root):
        self.seed = seed
        self.folder = run_root / f"seed_{seed}"
        self.lock = threading.Lock()
        self.rng = np.random.default_rng(20261125 + seed)
        self.env = Environment("space_invaders", sticky_action_prob=0.1)
        self.env.seed(20261125 + seed)
        self.env.reset()
        self.episode = 1
        self.game_step = 0
        self.score = 0.0
        self.done = False
        self.checkpoint_mtime = None
        self.checkpoint_steps = 0
        self.weights = None

    def load_policy(self):
        path = self.folder / "latest.pt"
        mtime = path.stat().st_mtime_ns
        if mtime == self.checkpoint_mtime:
            return
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        source = checkpoint["actor"]
        self.weights = {
            "W1": source["net.0.weight"].detach().numpy().T.copy(),
            "b1": source["net.0.bias"].detach().numpy().copy(),
            "W2": source["net.2.weight"].detach().numpy().T.copy(),
            "b2": source["net.2.bias"].detach().numpy().copy(),
            "W3": source["net.4.weight"].detach().numpy().T.copy(),
            "b3": source["net.4.bias"].detach().numpy().copy(),
        }
        self.checkpoint_steps = checkpoint["completed_updates"] * 8192
        self.checkpoint_mtime = mtime

    def frame(self):
        with self.lock:
            self.load_policy()
            if self.done:
                self.env.reset()
                self.episode += 1
                self.game_step = 0
                self.score = 0.0
                self.done = False
            active = observation_indices(self.env.state(), self.env.env.channels,
                                         CHANNELS["space_invaders"])
            action = policy_action(self.weights, active, self.rng)
            reward, done = self.env.act(action)
            self.score += float(reward)
            self.game_step += 1
            self.done = bool(done or self.game_step >= 10000)
            training_steps = recent_steps(self.folder / "train.log", self.checkpoint_steps)
            return {
                "seed": self.seed, "episode": self.episode, "game_step": self.game_step,
                "score": self.score, "done": self.done,
                "action": action, "action_name": ACTION_NAMES[action],
                "channels": self.env.state().astype(np.uint8).tolist(),
                "checkpoint_steps": self.checkpoint_steps,
                "training_steps": training_steps,
                "completed": (self.folder / "complete.json").exists(),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    spectators = {seed: Spectator(seed, args.run_root) for seed in (1, 2)}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/":
                data, mime, status = PAGE.encode(), "text/html; charset=utf-8", 200
            elif url.path == "/training":
                data, mime, status = TRAINING_PAGE.encode(), "text/html; charset=utf-8", 200
            elif url.path == "/api/training":
                try:
                    seed = int(parse_qs(url.query).get("seed", ["1"])[0])
                    if seed not in spectators:
                        raise ValueError("seed must be 1 or 2")
                    data = (args.run_root / f"seed_{seed}" / "live_frame.json").read_bytes()
                    status = 200
                except (ValueError, OSError) as error:
                    data, status = json.dumps({"error": str(error)}).encode(), 400
                mime = "application/json; charset=utf-8"
            elif url.path == "/api/frame":
                try:
                    seed = int(parse_qs(url.query).get("seed", ["1"])[0])
                    data = json.dumps(spectators[seed].frame()).encode()
                    status = 200
                except (ValueError, KeyError, OSError, RuntimeError) as error:
                    data = json.dumps({"error": str(error)}).encode()
                    status = 400
                mime = "application/json; charset=utf-8"
            else:
                data, mime, status = b"Not found", "text/plain", 404
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Watching checkpoints at http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
