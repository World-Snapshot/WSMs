
## Format Core Domain [0-31]

### wsg vae
- 0-2: RGB

### wsg vae
- 3: Alpha value

### wsg vae
- 4: Grayscale image




## Depth and Spatial Geometry Domain [32-63]

### wsg vae
- 32: Relative depth value (0-1)
- 33: Relative depth confidence (In fact, only a very small number of jobs offer this, and we are not using it for the time being.)

### wsg vae
- 34: Absolute depth value (meters)
- 35: Absolute depth confidence (In fact, only a very small number of jobs offer this, and we are not using it for the time being.)

### wsg vae
- 36-38: Surface normal vector (X,Y,Z)

### wsg vae
- 39: Curvature (For a sphere, each of its positions has a curvature. It's somewhat like the 3D version of a boundary.)

### wsg vae
- 40: Edge （False -1, True 1; -1 indicates the area of the mask, and this area is not taken into consideration at all.）

### wsg vae
- 41: optical flow dx
- 42: optical flow dv



## Pose and Human Information Domain [64-110]
### 2D skeletal human (DWPose) [64-71] (Necessary)

### wsg vae
- 64: Person ID (-1 or ID number)
    - 65: Keypoint ID (about 0-134, find this keypoint based on the xy coordinates, and then mark which key point it is.)
        - 66: Score (The body score serves as an index. One person occupies 18 index values, and then the second person increments. The remaining scores represent the degree of confidence.)



### 3D mesh human (SMPLer-X) [72-94] (Necessary)

### wsg vae
- 72: Person ID (from bbox_id in meta dir)
    - 73: Joint ID (global_orient + left_hand_pose + right_hand_pose + jaw_pose + leye_pose + reye_pose + transl)

### wsg vae
        - 74: Joint rotations or transl x
        - 75: Joint rotations or transl y
        - 76: Joint rotations or transl z

### wsg vae
- 77: betas: 0 epression: 1
    - 78-87: the value of betas/epression

### wsg vae
- 88: bbox: If the vertex is set to 1, then the absence of a vertex is naturally -1.
- 89: bbox_mmdet: If the vertex is set to 1, then the absence of a vertex is naturally -1.

### wsg vae
- 90: vertices (vf system) id
- 91: vertices (vf system) x: it is actually the coordinates that have been calibrated based on the predicted translation information.
- 92: vertices (vf system) y: it is actually the coordinates that have been calibrated based on the predicted translation information.
- 93: vertices (vf system) z: it is actually the coordinates that have been calibrated based on the predicted translation information.


### 3D skeletal human (Mediapipe Pose) [95-102] (Necessary)

### wsg vae
- 95: Person ID (-1 or ID number)
    - 96: Joint ID (Find this keypoint based on the xy coordinates, and then mark which key point it is.)
        - 97: z: relative depth: Positive values indicate proximity to the camera, while negative values indicate distance from the camera.
        - 98: Visibility confident



### Human segmentation [103-110] (Optional)

### wsg vae
- 103: class value (General by Mediapipe seg)



## Object/Animal Information Domain [111-125] (Necessary)

### wsg vae
  - 111: The mapped category/model ID

### wsg vae
  - 112: Class Value

### wsg vae
  - When Class Value = 1 (Vertex Data)
    - 113: Vertex ID - vertex index
    - 114: Vertex X coordinate
    - 115: Vertex Y coordinate
    - 116: Vertex Z coordinate
    - 117: Skinning weights (4 weights packed)
    - 118: Bone indices (4 bone IDs packed)
    - 119: Unused
    - 120: UV coordinate U
    - 121: UV coordinate V

  When Class Value = 2 (Face Data)
    - 113: Face ID - face index
    - 114: Vertex 1 index
    - 115: Vertex 2 index
    - 116: Vertex 3 index
    - 117-121: Unused

  When Class Value = 3 (Joint Data)
    - 113: Joint ID - joint index
    - 114: Joint X coordinate
    - 115: Joint Y coordinate
    - 116: Joint Z coordinate
    - 117: Skeleton metadata (stored only in first joint)
    - 118: Parent joint ID
    - 119-121: Unused

  When Class Value = 4 (Bone Data)
    - 113: Bone ID - bone index
    - 114: Start joint ID
    - 115: End joint ID
    - 116: Bone length
    - 117-121: Unused
  - 122-125: Reserved for future object properties


## Physics Properties Domain [126-142] (Necessary)

### wsg vae
  - 126: relative solidVolume (solid volume from metadata.csv) (normalized 0-100)
### wsg vae
  - 127: relative surfaceVolume (surface volume from metadata.csv)
### wsg vae
  - 128: relative weight (relative weight from metadata.csv)
### wsg vae
  - 129: supportSurfaceArea (relative support surface area from metadata.csv)
### wsg vae
  - 130: relative staticFrictionForce (relative static friction force from metadata.csv)
### wsg vae
  - 131: Physics material_id (material ID, single material)
### wsg vae
  - 132: material_confidence (material confidence/probability based on ratio from materials.csv)
### wsg vae
  - 133: density (density from densities.csv) (kg/L)
### wsg vae
  - 134: staticFrictionCoeff (static friction coefficient from densities.csv)
### wsg vae
  - 135: isContainerLike (container flag from metadata.csv, 0/1)
### wsg vae
  - 136: Object/Thing category id




## Material & Light Properties Domain [143-174]

### Tradition and Control Optics [143-160] (Optional)

### wsg vae
- 143: per-pixel material region ID
### wsg vae
- 144-146: Ambient light color (map Ka)
### wsg vae
- 147-149: Diffuse color (map Kd)
### wsg vae
- 150: Opacity (d)
### wsg vae
- 151-153: Specular reflection color (Ks)
### wsg vae
- 154: Specular exponent/shininess (Ns)
- 155-157: Irradiance RGB, normalized [0,1] or log-normalized HDR


### PBR Material [161-174] (Necessary) (Encodable control domain)


### wsg vae
- 161-163: Basecolor / Albedo - Base color map
### wsg vae
- 164-166: Diffuse - Diffuse map
### wsg vae
- 36-38: Normal - Normal map
### wsg vae
- 167: Height - Height map
### wsg vae
- 168: Roughness - Roughness map
### wsg vae
- 169: Metallic - Metallic map
### wsg vae
- 170-172: Specular - Specular map
### wsg vae
- 173: Opacity - Opacity map (when needed, e.g., for surfaces with holes)
### wsg vae
- 174: Reserved for future properties


## Sound And Language Properties Domain [175-206]
### Video Original Audio [175-184] (Necessary)
### wsg vae
- 175-184: The original/background audio of the video. If there is none, it's fine. For the specific implementation, please refer to the next section "WSV".

### Video Description Audio [185-194] (Necessary)
### wsg vae
- 185-194: Video caption describes audio. For the specific implementation, please refer to the next section "WSV".

### Extension [195-206]
### wsg vae
- 195: Semantic Segmentation


## Abstract Information Domain [207-255]

### Wireless Sensing [207-255]

#### LiDAR Point Cloud [207-210] (Necessary)

### wsg vae
- 207: Point Coordinate X (camera coordinate system X, in meters)
- 208: Point Coordinate Y (camera coordinate system Y, in meters)
- 209: Point Coordinate Z (camera coordinate system Z, in meters)


#### mmWave Radar Point Cloud [211-216] (Necessary)
### wsg vae
- 211: Point Coordinate X (camera coordinate system X, in meters)
- 212: Point Coordinate Y (camera coordinate system Y, in meters)
- 213: Point Coordinate Z (camera coordinate system Z, in meters)
- 214: Doppler Velocity (Doppler velocity d, raw units)
- 215: Signal Intensity (signal intensity I, raw units)


#### WiFi CSI Processed [217-235] (Necessary)
### wsg vae
- 217-226: CSI PCA Feature 1-10

