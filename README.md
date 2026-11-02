# Neonetra
NEONETRA -- Vision Redefined
NEONETRA is a chest-mounted assistive wearable designed to support
people with visual impairments through hands-free environmental
awareness, navigation guidance, text reading, and emergency assistance.

The goal is to make everyday movement and access to information safer,
more convenient, and more independent.
The Problem
Traditional white canes mainly detect obstacles at ground level and may
not provide protection from chest-level or overhanging obstacles. Basic
smart canes often provide simple warnings without explaining what an
obstacle is or guiding the user toward a safer route.

Smartphone-based assistance can also be inconvenient when users need to
hold and operate a phone while moving.
Our Solution
NEONETRA combines camera-based computer vision, Time-of-Flight (ToF)
depth sensing, audio output, and haptic feedback in a wearable device.
Key Features
- Autonomous navigation guidance: Uses computer vision, YOLO-based
  segmentation, and depth information to identify obstacles and
  determine a safer available path. Directional haptic feedback from
  vibration motors guides the user toward the clearer route.
- AI scene description: Provides spoken descriptions of the user's
  surroundings, including nearby objects and obstacles.
- OCR and text reading: Reads visible text such as documents,
  notes, and signboards through audio output.
- Currency-note recognition: Helps users identify currency notes.
- Dedicated emergency control: A separate physical button
  activates the device's emergency functionality.
- Hands-free interaction: Dedicated physical buttons let users
  access different functions without continuously holding a
  smartphone.
System Architecture
The project follows a thin-client approach:
1. The wearable captures visual and ToF depth information.
2. The captured information is sent to a local smartphone or laptop for
   processing through WebSockets.
3. Computer-vision and AI processing support scene understanding and
   safe-path identification.
4. Audio feedback communicates descriptions and recognized text.
5. Haptic motors provide directional navigation guidance and alerts.
The project presentation also describes a ToF-based hardware emergency
response for drop-offs and extremely close-range threats.
Technology
The documented design includes:
- Camera-based visual input
- Time-of-Flight (ToF) depth sensors
- YOLOv8 segmentation for safe-area identification
- WebSockets for communication between the wearable and local
  processing device
- Gemini 1.5 Flash API for scene-description and OCR-related
  assistance
- L9110S motor driver
- Haptic vibration motors
- Bluetooth audio output
Intended Users
NEONETRA is designed for visually impaired individuals navigating places
such as:
- Indoor environments
- University campuses
- Public spaces and crowded areas
Prototype Status
The project is under development. The project presentation describes the
prototype as 50% completed. Features and performance should be
validated through further integration, testing, and feedback from users.
Future Roadmap
- Finalize and improve the wearable hardware enclosure.
- Test the device in real-world environments with visually impaired
  users.
- Improve reliability, usability, and power performance.
- Seek feedback and pilot partnerships with NGOs and institutions for
  visually impaired individuals.
- Refine manufacturing costs and prepare for wider deployment.
Vision
NEONETRA aims to make navigation and access to everyday information
safer, smarter, and more independent.
Note: NEONETRA is an assistive-technology prototype under development.
It should not be treated as a replacement for a white cane, guide dog,
or other established mobility aid. Users should follow appropriate
safety practices while testing the device.
