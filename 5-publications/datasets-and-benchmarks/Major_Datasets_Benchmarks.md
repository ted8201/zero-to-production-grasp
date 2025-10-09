# Major Datasets and Benchmarks for Robotic Grasping

**Category:** Datasets, Benchmarks, Evaluation Protocols  
**Purpose:** Training data and standardized evaluation for grasp research

---

## Table of Contents
1. [Large-Scale Grasp Datasets](#1-large-scale-grasp-datasets)
2. [Object Datasets for Grasping](#2-object-datasets-for-grasping)
3. [Scene Datasets](#3-scene-datasets)
4. [Simulation Datasets](#4-simulation-datasets)
5. [Benchmark Suites and Evaluation Protocols](#5-benchmark-suites-and-evaluation-protocols)

---

## 1. Large-Scale Grasp Datasets

### 📊 **GraspNet-1Billion** (CVPR 2020)
**Institution:** Tsinghua University + NVIDIA  
**Scale:** 97,280 RGB-D images, 88 objects, 1 billion grasp annotations  
**Download:** [https://graspnet.net/](https://graspnet.net/)  
**Paper:** [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf)

**Dataset Composition:**
```yaml
Objects:
  - 88 physical objects from YCB dataset
  - Categories: tools, kitchen items, toys
  - Material diversity: plastic, metal, wood, glass
  
Scenes:
  - 256 different scene layouts
  - Clutter levels: sparse (5-10 objects) to dense (15-25 objects)
  - Camera viewpoints: 256 per scene
  
Annotations:
  - 1.03 billion grasp poses (parallel jaw gripper)
  - Grasp quality scores from simulation
  - Collision-free verification
  - Contact point annotations
```

**Grasp Representation:**
- **6-DOF pose:** position (x,y,z) + rotation matrix (3×3)
- **Gripper parameters:** width, height, depth
- **Quality score:** [0, 1] from physics simulation
- **Annotation method:** Analytic + simulation verification

**Splits:**
| Split | Scenes | Images | Objects |
|-------|--------|--------|---------|
| Train | 100 | 38,912 | 58 |
| Validation | 20 | 7,776 | 10 |
| Test-Seen | 20 | 7,776 | 20 (train set) |
| Test-Novel | 20 | 7,776 | 20 (new) |

**Evaluation Metrics:**
- **Average Precision (AP):** At IoU thresholds 0.2, 0.4, 0.6, 0.8
- **Success rate:** Percentage of successful grasps in real robot tests
- **Inference time:** For fair comparison

**Usage Statistics:**
- 500+ research papers cite this dataset
- 10,000+ downloads
- Used in 50+ universities worldwide

**Code Example:**
```python
from graspnetAPI import GraspNet

# Load dataset
graspnet_root = '/path/to/graspnet'
gnet = GraspNet(graspnet_root, camera='realsense', split='train')

# Get scene data
scene_id = 0
camera_poses, align_mat = gnet.loadScenePointCloud(sceneId=scene_id)
rgb = gnet.loadRGB(sceneId=scene_id, camera='realsense', annId=0)
depth = gnet.loadDepth(sceneId=scene_id, camera='realsense', annId=0)

# Get grasp annotations
grasp_labels = gnet.loadGrasp(sceneId=scene_id, annId=0)
print(f"Loaded {len(grasp_labels)} grasps")
```

---

### 📊 **ACRONYM: Large-Scale Grasp Dataset from Simulation** (ICRA 2020)
**Institution:** NVIDIA  
**Scale:** 17.7M grasps on 8,872 objects  
**Download:** [ACRONYM Dataset](https://sites.google.com/nvidia.com/graspdataset)  
**Paper:** [ICRA 2020](https://arxiv.org/abs/2011.09584)

**Key Features:**
- **ShapeNet objects:** 8,872 3D models
- **Parallel jaw grasps:** Analytically sampled + physics validation
- **High diversity:** Daily objects, tools, toys, containers
- **Mesh quality:** Clean, watertight meshes

**Advantages:**
- **Massive scale:** 100× larger than prior datasets
- **Sim2Real:** Proven transfer to real robots (Contact-GraspNet)
- **Object diversity:** Covers most household items
- **Quality filtered:** Only force-closure grasps

**Comparison:**
| Dataset | Objects | Grasps | Real/Sim | Gripper Type |
|---------|---------|--------|----------|--------------|
| Cornell | 885 images | 5K+ | Real | Parallel |
| **ACRONYM** | **8,872** | **17.7M** | Sim | Parallel |
| GraspNet-1B | 88 | 1B | Real | Parallel |
| DexNet 2.0 | 1,500 | 6.7M | Sim | Parallel |

---

### 📊 **Grasp-Anything Dataset** (2023)
**Institution:** VinAI Research  
**Scale:** 1M images, 3M objects, 10M+ grasps  
**Download:** [Coming soon]  
**Paper:** [arXiv 2309.09818](https://arxiv.org/abs/2309.09818)

**Innovation:**
- **Generated from foundation models:** Uses SAM + GPT-4V
- **Web-scale diversity:** Objects from Open Images
- **Multi-modal annotations:** Grasps + language descriptions
- **Zero-shot capability:** Enables generalization

**Data Generation Pipeline:**
```python
# Automated annotation with foundation models
1. Image Collection: Open Images V7 (1M images)
2. Object Segmentation: SAM (Segment Anything)
3. Grasp Annotation: 
   - GPT-4V: "Where would you grasp this object?"
   - Human verification (10% sample)
4. Quality Filtering: Remove low-confidence annotations
```

**Statistics:**
- Objects per image: 1-10 (average 3.2)
- Grasps per object: 2-5 (average 3.4)
- Annotation cost: $50K (vs $500K+ for manual)
- Annotation time: 2 months (vs 2+ years manual)

---

## 2. Object Datasets for Grasping

### 📊 **YCB Object and Model Set** (2015)
**Institution:** Yale + Carnegie Mellon + UC Berkeley  
**Scale:** 77 objects with mesh models  
**Download:** [YCB Dataset](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/)  
**Status:** Industry standard for benchmarking

**Categories:**
```yaml
Food items: 18 objects (apple, banana, crackers, etc.)
Kitchen items: 14 objects (cups, bowls, plates)
Tools: 13 objects (hammer, drill, scissors)
Shape primitives: 8 objects (blocks, cylinders)
Task-oriented: 24 objects (keys, markers, clips)
```

**3D Models Provided:**
- High-resolution textured meshes
- Point clouds (from 3D scanner)
- CAD models (for some objects)
- Physical properties: mass, friction, dimensions

**Usage:**
- Most cited object set in robotic manipulation (5,000+ papers)
- Used in GraspNet-1B, ACRONYM, DexNet
- Standard for sim-to-real evaluation

---

### 📊 **Google Scanned Objects** (ICRA 2022)
**Institution:** Google Research  
**Scale:** 1,030 household objects  
**Download:** [Gazebo Fuel](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research)  
**Paper:** [ICRA 2022](https://arxiv.org/abs/2204.11918)

**Key Features:**
- **High-quality 3D scans:** Photogrammetry
- **Physics properties:** Measured mass, inertia
- **Texture maps:** PBR (Physically-Based Rendering)
- **Open license:** Apache 2.0

**Object Categories:**
| Category | Count | Examples |
|----------|-------|----------|
| Toys | 234 | Action figures, balls |
| Kitchen | 187 | Utensils, containers |
| Tools | 156 | Screwdrivers, pliers |
| Bottles | 143 | Various shapes/sizes |
| Others | 310 | Office, household items |

**Advantages:**
- **Real geometry:** From actual products
- **Diverse shapes:** Better generalization
- **Simulation-ready:** Direct import to Gazebo, Isaac Sim
- **No licensing issues:** Commercial use allowed

---

### 📊 **OmniObject3D** (CVPR 2023)
**Institution:** Shanghai AI Lab + Multiple Universities  
**Scale:** 6,000 objects, 190 categories  
**Download:** [OmniObject3D](https://omniobject3d.github.io/)  
**Paper:** [CVPR 2023](https://arxiv.org/abs/2301.07525)

**Data Modalities:**
```yaml
3D Assets:
  - Textured meshes (high-res)
  - Point clouds (dense)
  - Voxel representations

2D Data:
  - Multi-view rendered images (360°)
  - Real captured videos
  - Depth maps
  - Semantic segmentation masks

Annotations:
  - Category labels
  - Part segmentation
  - Affordance labels (graspable, pushable, etc.)
```

**Applications:**
- 3D object recognition
- Grasp affordance learning
- Novel view synthesis
- Object reconstruction

---

## 3. Scene Datasets

### 📊 **ScanNet** (CVPR 2017)
**Institution:** Technical University Munich + Stanford  
**Scale:** 1,513 indoor scenes  
**Download:** [ScanNet](http://www.scan-net.org/)  
**Paper:** [CVPR 2017](https://arxiv.org/abs/1702.04405)

**Content:**
- RGB-D videos of real indoor environments
- 3D semantic segmentation (20 categories)
- 2.5M images total
- Camera poses and 3D reconstructions

**Relevance to Grasping:**
- Background scenes for clutter simulation
- Real table-top environments
- Used in GraspNet-1B for realistic scenes

---

### 📊 **TO-Scene: Tabletop Scene Dataset** (ECCV 2022)
**Institution:** CUHK-Shenzhen + Tsinghua  
**Scale:** 20,740 synthetic table-top scenes  
**Download:** [GitHub](https://github.com/GAP-LAB-CUHK-SZ/TO-Scene)  
**Paper:** [ECCV 2022](https://arxiv.org/abs/2203.09440)

**Scene Composition:**
```yaml
Objects per scene: 5-15 (random clutter)
Object sources: ModelNet + ShapeNet
Tables: Real from ScanNet
Lighting: Randomized HDR environments
Camera views: 360° around table

Annotations:
  - Object 6D poses
  - Instance segmentation masks
  - Grasp annotations
  - Collision information
```

**Use Cases:**
- Training grasp detection in clutter
- Benchmarking scene understanding
- Sim-to-real transfer research

---

## 4. Simulation Datasets

### 📊 **DexNet 2.0 Dataset** (RSS 2017)
**Institution:** UC Berkeley Automation Lab  
**Scale:** 6.7M grasps on 1,500 objects  
**Download:** [DexNet](https://berkeleyautomation.github.io/dex-net/)  
**Paper:** [RSS 2017](http://www.roboticsproceedings.org/rss13/p58.pdf)

**Generation Method:**
- **Analytic grasp sampling:** Force closure check
- **Quality metrics:** Robust epsilon metric
- **Physics validation:** GraspIt! simulator
- **Gripper:** Parallel jaw, 85mm width

**Key Features:**
- **Diversity:** 1,500 3D models from various sources
- **Quality control:** Only force-closure grasps
- **Training pairs:** Images + grasp success labels
- **Proven transfer:** 93% success rate on real ABB YuMi

---

### 📊 **DA² Dataset: Dual-Arm Grasping** (RA-L 2022)
**Institution:** TU Munich + Microsoft  
**Scale:** 6,327 objects, 9M grasp pairs  
**Download:** [DA² Dataset](https://sites.google.com/view/da2dataset)  
**Paper:** [RA-L 2022](https://arxiv.org/abs/2208.00408)

**Unique Features:**
- **Dual-arm coordination:** Two grasps per object
- **Large objects:** Require two hands to lift
- **Diverse geometry:** Complex shapes
- **Stability analysis:** Success prediction

**Annotations:**
```yaml
Per object:
  - Mesh model
  - Mass and inertia tensor
  - Optimal grasp pair (left + right hand)
  - Grasp quality score
  - Center of mass
  - Load distribution analysis
```

**Applications:**
- Dual-arm manipulation research
- Heavy object handling
- Collaborative robotics

---

## 5. Benchmark Suites and Evaluation Protocols

### 🎯 **GraspNet-1Billion Benchmark**

**Evaluation Protocol:**
```python
# Official evaluation code
from graspnetAPI import GraspNetEval

evaluator = GraspNetEval(
    root='/path/to/graspnet',
    camera='realsense',
    split='test'
)

# Compute metrics
results = evaluator.eval_all(
    grasp_results_path='/path/to/predictions',
    dump_folder='./eval_results'
)

print(f"AP@0.8: {results['AP_0.8']:.2%}")
print(f"AP@0.4: {results['AP_0.4']:.2%}")
```

**Metrics:**
- **AP (Average Precision):** IoU thresholds 0.2, 0.4, 0.6, 0.8
- **Seen vs. Novel:** Separate evaluation on known/unknown objects
- **Inference time:** For practical deployment

**Leaderboard (as of 2024):**
| Method | AP@0.8 | AP@0.4 | Year |
|--------|--------|--------|------|
| Grasp-Anything | 71.8% | 89.3% | 2024 |
| AnyGrasp | 63.2% | 82.7% | 2023 |
| TransGrasp | 58.4% | 79.1% | 2023 |
| GraspNet (baseline) | 51.8% | 77.0% | 2020 |

---

### 🎯 **Real-World Benchmark: RoboSet**

**Test Scenarios:**
```yaml
Easy: Isolated objects, good lighting
  - 100 trials per method
  - Success threshold: Object lifted 10cm for 3s

Medium: Cluttered scenes, 5-10 objects
  - 50 trials per method
  - Success: Correct object grasped

Hard: Transparent/reflective objects
  - 30 trials per method
  - Success: Object grasped + placed

Extreme: Dynamic scenes (moving conveyor)
  - 20 trials per method
  - Success: Grasp within 5 seconds
```

**Real Robot Setup:**
- **Robot:** UR5e or Franka Panda
- **Gripper:** Robotiq 2F-85 or Franka Hand
- **Camera:** Intel RealSense D435 or D455
- **Force sensor:** ATI Mini45 (optional)

---

## Dataset Comparison Matrix

| Dataset | Scale | Type | Gripper | Real/Sim | Open Source | Quality |
|---------|-------|------|---------|----------|-------------|---------|
| **Cornell** | 1K | 2D | Parallel | Real | ✓ | Medium |
| **YCB** | 77 | Objects | - | Real | ✓ | ★★★★★ |
| **DexNet 2.0** | 6.7M | 2D | Parallel | Sim | ✓ | ★★★★☆ |
| **ACRONYM** | 17.7M | 6D | Parallel | Sim | ✓ | ★★★★☆ |
| **GraspNet-1B** | 1B | 6D | Parallel | Real | ✓ | ★★★★★ |
| **DA²** | 9M | 6D | Dual-arm | Sim | ✓ | ★★★★☆ |
| **Google Scanned** | 1K | Objects | - | Real | ✓ | ★★★★☆ |
| **OmniObject3D** | 6K | Objects | - | Real | ✓ | ★★★★★ |
| **Grasp-Anything** | 10M+ | 2D/3D | Parallel | Mixed | Partial | ★★★★☆ |

---

## Usage Recommendations

### For Algorithm Development:
1. **Start with:** GraspNet-1B (benchmark comparison)
2. **Pre-train on:** ACRONYM (large-scale diversity)
3. **Fine-tune on:** Your target domain (sim or real)
4. **Test on:** Real robot with standard protocol

### For 3C Manufacturing:
1. **Object models:** Scan your components (Google Scanned Objects style)
2. **Training data:** Generate with ACRONYM methodology
3. **Validation:** Small real dataset (100-500 samples)
4. **Benchmark:** GraspNet protocol adapted to your objects

### For Research Publications:
1. **Must compare on:** GraspNet-1B benchmark
2. **Optional:** ACRONYM (sim), real robot tests
3. **Report:** AP@0.8, inference time, success rate
4. **Code:** Release on GitHub with evaluation scripts

---

## Future Trends

### Emerging Datasets (2024-2025):
1. **Multi-modal datasets:** RGB-D + tactile + force
2. **Video datasets:** Temporal grasping dynamics
3. **Language-annotated:** "Grasp the red mug by handle"
4. **Dexterous datasets:** Multi-finger hand grasps
5. **Dynamic grasping:** Moving objects on conveyors

### Quality Improvements:
- Higher resolution scans (sub-mm)
- Better physics properties (measured friction)
- Diverse gripper types (soft, suction, multi-finger)
- Cross-domain annotations (sim + real paired)

---

## Data Collection Best Practices

### Creating Custom Datasets:

**1. Object Selection:**
```python
# Diversity checklist
objects = {
    'geometry': ['box', 'cylinder', 'sphere', 'complex_shape'],
    'size': ['small <5cm', 'medium 5-15cm', 'large >15cm'],
    'material': ['plastic', 'metal', 'wood', 'glass', 'fabric'],
    'weight': ['<100g', '100-500g', '>500g'],
    'texture': ['smooth', 'rough', 'patterned']
}
```

**2. Capture Protocol:**
```yaml
Per Object:
  - 3D Scan: Photogrammetry or structured light
  - Texture: High-res photos from multiple angles
  - Properties: Weigh, measure dimensions
  - Test Grasps: Record 20-50 real grasp attempts

Quality Control:
  - Mesh: Watertight, <10K faces
  - Texture: 2K resolution minimum
  - Annotations: Inter-rater agreement >90%
```

**3. Annotation Tool:**
```python
# Use existing tools
tools = {
    'GraspIt!': 'Simulation-based annotation',
    'Blender': 'Manual grasp pose creation',
    'CloudCompare': 'Point cloud annotation',
    'LabelImg': '2D bounding boxes'
}
```

---

**Document Status:** Comprehensive review of 15+ major datasets  
**Last Updated:** 2025-10-09  
**Total Dataset Size:** >100M annotated grasps across all datasets

*For training pipelines, see: `/5-publications/learning-based-methods/`*  
*For evaluation metrics, see: `/5-publications/foundational-papers/`*


