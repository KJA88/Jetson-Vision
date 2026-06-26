DHRAS Start / Stop Manual
Current Stable System
Main dashboard:
http://192.168.1.17:8080
Direct camera streams:
Front yard:
http://192.168.1.17:8081/stream/frontyard

Backyard:
http://192.168.1.17:8081/stream/backyard

Indoor:
http://192.168.1.17:8081/stream/indoor

Health check:
http://192.168.1.17:8081/health
Current working files:
vision_service.py  = cameras, YOLO, PTZ, streams on port 8081
dashboard.py       = main dashboard on port 8080
start.sh           = starts DHRAS
stop.sh            = stops DHRAS
Do not use right now:
dashboard_v2.py
http://192.168.1.17:8083
stream_proxy.py
camera_worker.py
FY_MON.py
BY_MON_TRK.py
ID_MON_TRK.py
Frontyard_Logging.py
Backyard_Logging.py
Indoor_tracker_v2.py
Zoom buttons were removed because ONVIF zoom was unreliable on the current PTZ cameras.
________________________________________
Normal Start
Use this for normal daily startup:
cd ~/robotics/jetson-vision
./start.sh
Then open:
http://192.168.1.17:8080
________________________________________
Normal Stop
Use this to stop DHRAS:
cd ~/robotics/jetson-vision
./stop.sh
________________________________________
Quick Check If Something Looks Wrong
Use this if the dashboard looks wrong, cameras say offline, or streams do not appear:
cd ~/robotics/jetson-vision

ps aux | grep -E "vision_service.py|dashboard.py" | grep -v grep
ss -ltnp | grep -E ':8080|:8081'
curl -s http://127.0.0.1:8080/api/cameras/status
Expected good status:
{"backyard":{"mode":"ACTIVE","online":true},"frontyard":{"mode":"ACTIVE","online":true},"indoor":{"mode":"ACTIVE","online":true}}
________________________________________
Full Start + Verify
Use this when you want to restart everything and confirm all services, ports, statuses, and links.
cd ~/robotics/jetson-vision

echo "=== STOPPING DHRAS ==="
./stop.sh

sleep 3

echo
echo "=== STARTING DHRAS ==="
./start.sh

echo
echo "=== RUNNING PROCESSES ==="
ps aux | grep -E "vision_service.py|dashboard.py" | grep -v grep

echo
echo "=== PORTS ==="
ss -ltnp | grep -E ':8080|:8081'

echo
echo "=== DASHBOARD CAMERA STATUS ==="
curl -s http://127.0.0.1:8080/api/cameras/status
echo

echo
echo "=== DIRECT VISION STATUS ==="
curl -s http://127.0.0.1:8081/status/frontyard
echo
curl -s http://127.0.0.1:8081/status/backyard
echo
curl -s http://127.0.0.1:8081/status/indoor
echo

echo
echo "=== LINKS ==="
echo "Dashboard:  http://192.168.1.17:8080"
echo "Frontyard:  http://192.168.1.17:8081/stream/frontyard"
echo "Backyard:   http://192.168.1.17:8081/stream/backyard"
echo "Indoor:     http://192.168.1.17:8081/stream/indoor"
echo "Health:     http://192.168.1.17:8081/health"
Expected good result:
8080 listening = dashboard.py
8081 listening = vision_service.py

frontyard = ACTIVE true
backyard  = ACTIVE true
indoor    = ACTIVE true
________________________________________
Stable Backup Location
Known-good backups are stored in:
~/robotics/jetson-vision/SAFE_BACKUPS
Before changing code, make a backup first:
cd ~/robotics/jetson-vision
cp dashboard.py dashboard.py.before_change
cp vision_service.py vision_service.py.before_change

