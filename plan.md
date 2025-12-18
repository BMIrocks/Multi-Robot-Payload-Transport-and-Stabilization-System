# DSAC Bots - Coordinated Payload Transport System

## Project Overview

A multi-robot system consisting of 3-4 wheeled carriers that coordinate to transport payloads on a shared platform (plywood sheet). Each carrier is equipped with sensors and cameras, communicating through a centralized master-slave architecture to maintain synchronized movement and payload stability.

---

## System Architecture

### Hardware Components

**Carriers (4 units):**
- 4-wheeled chassis with motor drivers (L293D/L298 with encoders)
- Load cells on each bot to measure weight distribution
- IMU for tilt detection and deviation monitoring
- Camera for path identification and obstacle detection
- Depth sensor + IR (ultrasonic optional)
- Local MCU: **ESP32** | **STM32** | **Raspberry Pi**

**Shared Platform:**
- Plywood sheet mounted on top of all 4 carriers
- Payload placement area with load sharing capability

**Central Coordinator:**
- Master unit (Raspberry Pi preferred)
- WiFi-mesh communication with all carriers
- Processes 3rd party tracking software inputs
- Coordinates movement commands

---

## Key Features

### 1. Load Distribution & Balancing
- **Load cells** on each carrier measure real-time weight distribution
- **Dynamic speed adjustment** using PID control to balance load
- Ensures even distribution across all 4 carriers

### 2. Payload Stability
- **PID control** with leveling and load sharing
- **Pistons** for lifting/lowering and shock absorption
- Prevents payload from tipping during turns or uneven terrain

### 3. Obstacle & Ramp Detection
- **Option A:** LIDAR + PIR sensor
- **Option B:** 2 cameras with wider view angle
- **Option C:** Ultrasonic sensor (up-down scanning)
- Detects drop zones, ramps, and obstacles

### 4. Computer Vision Integration
- Uses **3rd party software** provided by Prof
- Identifies paths, drop zones, and depths
- Two possible modes:
  - Floor layout-based tracking
  - Surroundings view-based navigation
- Software augments places using given framework

### 5. Synchronization & Coordination
- **Master-slave architecture** with central coordinator
- WiFi-mesh for real-time communication
- Master dynamically adjusts speeds and direction of all carriers
- Handles edge cases:
  - One bot encounters obstacle first
  - Network connectivity issues
  - Command skipped/delayed
  - Asymmetric turning (payload could fall)

---

## Control System

### PID Control Implementation
- **Purpose:** Dynamic speed and direction adjustment
- **Inputs:** 
  - Load cell readings (weight distribution)
  - IMU data (tilt angles)
  - Computer vision data (path alignment)
- **Outputs:** 
  - Individual motor speeds for each carrier
  - Synchronized movement commands

### Motor Control
- **Motor Drivers:** L293D or L298 (with encoder feedback)
- **PWM Control** for speed regulation
- **Encoder feedback** for precise movement tracking

---

## Communication Architecture

### Network Topology
```
    [Central Master]
         |
    WiFi-Mesh
    /  |  |  \
   C1  C2 C3 C4
   (Carriers)
```

### Protocols
- **WiFi-mesh** for carrier-to-master communication
- **Master broadcasts** synchronized commands
- **Error handling:** Retry mechanism for dropped commands

---

## Simulation & Testing

### Simulation Platforms
- **MATLAB Simulink** - System modeling and control simulation
- **Webots** - Robot simulation environment

### Test Scenarios
1. **Flat surface transport** - Basic synchronization test
2. **Rough terrain** - Shock absorption and stability test
3. **Ramp navigation** - Obstacle detection and coordination
4. **Turn coordination** - Payload stability during turns
5. **Single bot obstacle** - Edge case handling (3 bots move, 1 stops)
6. **Network failure** - Communication dropout recovery

---

## Implementation Phases

### Phase 1: Hardware Setup
- Assemble 4 carriers with motors and chassis
- Install sensors (load cells, IMU, cameras, depth sensors)
- Set up motor drivers and MCUs
- Mount plywood platform

### Phase 2: Software Development
- Implement PID control algorithms
- Develop master-slave communication protocol
- Integrate 3rd party vision software
- Build command synchronization logic

### Phase 3: Integration & Testing
- Test individual carrier movements
- Calibrate load cells and sensors
- Test coordinated movement without payload
- Test with payload on rough surfaces

### Phase 4: Optimization
- Fine-tune PID parameters
- Optimize network latency
- Add failsafe mechanisms
- Implement advanced edge case handling

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Uneven weight distribution** | Load cells + PID control for dynamic balancing |
| **Payload tipping on turns** | Synchronized speed control + tilt detection (IMU) |
| **Network latency/drops** | Command buffering + retry mechanism |
| **One bot hits obstacle** | Master detects deviation, adjusts other 3 bots accordingly |
| **Rough terrain stability** | Shock-absorbing pistons + suspension system |
| **Vision software integration** | Need to clarify inputs/outputs with Prof (floor vs surroundings mode) |

---

## Pending Clarifications

- [ ] Exact specifications of 3rd party tracking software
- [ ] Input format required by vision software
- [ ] Floor layout vs surroundings view - which mode?
- [ ] Payload weight range
- [ ] Maximum terrain roughness requirements

---

## Technologies

**Hardware:**
- ESP32/STM32/Raspberry Pi
- L293D/L298 Motor Drivers
- Load Cells, IMU, Cameras
- LIDAR/PIR/Ultrasonic sensors
- Pistons (optional for advanced stability)

**Software:**
- PID Control Algorithms
- WiFi-mesh networking
- Computer Vision (3rd party)
- MATLAB Simulink / Webots (simulation)

**Communication:**
- WiFi-mesh protocol
- Serial communication (MCU ↔ sensors)
- Camera interface

---

**Status:** Planning Phase  
**Last Updated:** December 13, 2025
