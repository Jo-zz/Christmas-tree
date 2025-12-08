
import React, { useState, useEffect, useRef, Suspense, useMemo } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, Stars, Float, Text, useTexture, Sparkles } from "@react-three/drei";
import { EffectComposer, Bloom, Vignette, Noise } from "@react-three/postprocessing";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// --- Configuration ---
const PARTICLE_COUNT = 5500; // Slightly denser tree
const TREE_HEIGHT = 16;
const TREE_RADIUS = 6.5;

// True Christmas Colors
const COLOR_PINE_DEEP = "#013220"; // Very Dark Green
const COLOR_PINE_VIBRANT = "#228B22"; // Forest Green
const COLOR_RED = "#D42426"; // Vibrant Red
const COLOR_GOLD = "#FFD700"; // Pure Gold
const COLOR_WHITE = "#FFFFFF"; 

const EXPLOSION_SPEED = 0.08;
const RETURN_SPEED = 0.05;

// --- Types ---
type InteractionState = "tree" | "exploded";

// --- Helper Math ---
const randomRange = (min: number, max: number) => Math.random() * (max - min) + min;

// --- Global Singletons for MediaPipe ---
// We use a global promise to prevent double-downloading WASM files in React Strict Mode
let visionTasksPromise: Promise<any> | null = null;

const getVisionTasks = () => {
    if (!visionTasksPromise) {
        visionTasksPromise = FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
        );
    }
    return visionTasksPromise;
};

// --- Components ---

/**
 * HandTracker: Handles Webcam and MediaPipe logic
 * Communicates interaction state AND hand position for rotation
 */
const HandTracker = ({ 
    onUpdateInteraction 
}: { 
    onUpdateInteraction: (isOpen: boolean, handX: number) => void 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [permissionGranted, setPermissionGranted] = useState(false);
  const [loading, setLoading] = useState(true);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const animationFrameRef = useRef<number>(0);
  const lastVideoTimeRef = useRef<number>(-1);
  const initRef = useRef(false);

  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;
    
    let isMounted = true;

    const setupMediaPipe = async () => {
      try {
        const vision = await getVisionTasks();
        
        if (!isMounted) return;

        if (!landmarkerRef.current) {
            landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 1
            });
        }
        
        if (isMounted) setLoading(false);
      } catch (err) {
        console.error("MediaPipe Init Error:", err);
        if (isMounted) setLoading(false);
      }
    };

    setupMediaPipe();

    return () => {
      isMounted = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const enableCam = async () => {
    if (!landmarkerRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.addEventListener("loadeddata", predictWebcam);
        setPermissionGranted(true);
      }
    } catch (err) {
      console.error("Camera Error:", err);
      alert("Camera access denied or unavailable. You can still view the tree!");
    }
  };

  const predictWebcam = () => {
    if (!landmarkerRef.current || !videoRef.current) return;

    // Check if video is actually ready
    if (videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
      animationFrameRef.current = requestAnimationFrame(predictWebcam);
      return;
    }

    let startTimeMs = performance.now();
    
    if (videoRef.current.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = videoRef.current.currentTime;
      try {
        const results = landmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
        
        if (results.landmarks && results.landmarks.length > 0) {
          const landmarks = results.landmarks[0];
          const wrist = landmarks[0];
          
          // 1. Calculate Openness
          const tips = [4, 8, 12, 16, 20];
          let totalDist = 0;
          tips.forEach(tipIdx => {
            const tip = landmarks[tipIdx];
            const dist = Math.sqrt(Math.pow(tip.x - wrist.x, 2) + Math.pow(tip.y - wrist.y, 2));
            totalDist += dist;
          });
          const avgDist = totalDist / 5;
          // Threshold for open hand vs fist
          const isOpen = avgDist > 0.35; 

          // 2. Get Hand X Position (0.0 - 1.0)
          const handX = wrist.x;

          onUpdateInteraction(isOpen, handX);
        } else {
          // No hand detected - default state
          onUpdateInteraction(false, 0.5);
        }
      } catch (e) {
        console.warn("Detection error:", e);
      }
    }
    
    animationFrameRef.current = requestAnimationFrame(predictWebcam);
  };

  return (
    <div className="absolute bottom-4 left-4 z-50 flex flex-col items-start gap-2">
      <video ref={videoRef} className="w-32 h-24 rounded-lg border-2 border-slate-700 bg-black object-cover opacity-80" autoPlay playsInline muted style={{ transform: 'scaleX(-1)' }} />
      {!permissionGranted && (
        <button 
          onClick={enableCam} 
          disabled={loading}
          className="bg-green-900/80 backdrop-blur-md border border-red-500 text-white px-4 py-2 rounded-full text-xs font-bold tracking-widest hover:bg-green-800 transition-all uppercase shadow-[0_0_15px_rgba(255,0,0,0.5)]"
        >
          {loading ? "Loading AI..." : "Start Magic"}
        </button>
      )}
      {permissionGranted && (
        <div className="text-[10px] text-yellow-100 bg-black/50 px-2 py-1 rounded backdrop-blur-sm border border-white/10">
          <div className="flex items-center gap-2"><span>🖐️</span> <span>Open: Explode</span></div>
          <div className="flex items-center gap-2"><span>✊</span> <span>Fist: Build Tree</span></div>
          <div className="flex items-center gap-2"><span>↔️</span> <span>Move Hand: Rotate</span></div>
        </div>
      )}
    </div>
  );
};

/**
 * Dreamy Falling Snow Effect
 */
const Snow = () => {
    const count = 1500;
    const mesh = useRef<THREE.InstancedMesh>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);
    
    const particles = useMemo(() => {
        const temp = [];
        for (let i = 0; i < count; i++) {
            const x = randomRange(-30, 30);
            const y = randomRange(0, 50);
            const z = randomRange(-30, 30);
            const speed = randomRange(0.02, 0.1); 
            const phase = Math.random() * Math.PI * 2; 
            const scale = randomRange(0.05, 0.12);
            temp.push({ x, y, z, speed, phase, scale, originalX: x });
        }
        return temp;
    }, []);

    useFrame((state) => {
        if (!mesh.current) return;
        const time = state.clock.getElapsedTime();

        particles.forEach((p, i) => {
            p.y -= p.speed;
            p.x = p.originalX + Math.sin(time * 0.5 + p.phase) * 1.0;

            if (p.y < -15) {
                p.y = 40;
                p.x = randomRange(-30, 30);
                p.originalX = p.x;
            }
            
            dummy.position.set(p.x, p.y, p.z);
            dummy.scale.set(p.scale, p.scale, p.scale);
            dummy.rotation.x = time * 0.2 + p.phase;
            dummy.updateMatrix();
            mesh.current!.setMatrixAt(i, dummy.matrix);
        });
        mesh.current.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh ref={mesh} args={[undefined, undefined, count]}>
            <dodecahedronGeometry args={[0.5, 0]} />
            <meshStandardMaterial 
              color={COLOR_WHITE} 
              transparent 
              opacity={0.7}
              blending={THREE.AdditiveBlending} 
              depthWrite={false}
              roughness={0.1}
            />
        </instancedMesh>
    );
};


/**
 * The Tree Particle System
 */
const ParticleTree = ({ 
    targetState, 
    rotationTarget 
}: { 
    targetState: InteractionState; 
    rotationTarget: number;
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const groupRef = useRef<THREE.Group>(null);
  
  // Generate Data
  const { positionsTree, positionsUniverse, colors } = useMemo(() => {
    const pTree = new Float32Array(PARTICLE_COUNT * 3);
    const pUniverse = new Float32Array(PARTICLE_COUNT * 3);
    const col = new Float32Array(PARTICLE_COUNT * 3);
    const colorObj = new THREE.Color();

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // --- Tree Shape (Spiral Cone) ---
      const t = i / PARTICLE_COUNT;
      const angle = t * Math.PI * 40; 
      const yTree = -TREE_HEIGHT/2 + t * TREE_HEIGHT;
      const rTree = (1 - t) * TREE_RADIUS; 
      
      const noiseR = randomRange(0.9, 1.1); 
      
      pTree[i * 3] = Math.cos(angle) * rTree * noiseR;
      pTree[i * 3 + 1] = yTree;
      pTree[i * 3 + 2] = Math.sin(angle) * rTree * noiseR;

      // --- Universe Shape ---
      const rUni = randomRange(10, 50);
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      
      pUniverse[i * 3] = rUni * Math.sin(phi) * Math.cos(theta);
      pUniverse[i * 3 + 1] = rUni * Math.sin(phi) * Math.sin(theta);
      pUniverse[i * 3 + 2] = rUni * Math.cos(phi);

      // --- Colors & Decor ---
      const rand = Math.random();
      let cHex;

      // Distribution:
      // 5% Top Gold Stars (Large)
      // 10% Red Ornaments (Medium)
      // 5% White Lights (Small)
      // 80% Green Foliage (Small/Base)
      
      if (rand > 0.95) cHex = COLOR_GOLD;    // Yellow Stars
      else if (rand > 0.85) cHex = COLOR_RED; // Red Ornaments
      else if (rand > 0.80) cHex = COLOR_WHITE; // Sparkle lights
      else if (rand > 0.40) cHex = COLOR_PINE_VIBRANT; // Lighter green
      else cHex = COLOR_PINE_DEEP; // Darker green base
      
      colorObj.set(cHex);
      
      // Slightly vary the greens for realism
      if (cHex === COLOR_PINE_VIBRANT || cHex === COLOR_PINE_DEEP) {
         colorObj.offsetHSL(randomRange(-0.04, 0.04), 0, 0);
      }

      col[i * 3] = colorObj.r;
      col[i * 3 + 1] = colorObj.g;
      col[i * 3 + 2] = colorObj.b;
    }

    return { positionsTree: pTree, positionsUniverse: pUniverse, colors: col };
  }, []);

  const currentPositions = useRef(new Float32Array(positionsTree));
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useFrame((state, delta) => {
    if (!meshRef.current || !groupRef.current) return;

    // Rotation from hand
    const targetRotY = (rotationTarget - 0.5) * Math.PI * 3; 
    groupRef.current.rotation.y += (targetRotY - groupRef.current.rotation.y) * 0.08;

    const step = targetState === "exploded" ? EXPLOSION_SPEED : RETURN_SPEED;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const idx = i * 3;
      
      const tx = targetState === "exploded" ? positionsUniverse[idx] : positionsTree[idx];
      const ty = targetState === "exploded" ? positionsUniverse[idx + 1] : positionsTree[idx + 1];
      const tz = targetState === "exploded" ? positionsUniverse[idx + 2] : positionsTree[idx + 2];

      const cx = currentPositions.current[idx];
      const cy = currentPositions.current[idx + 1];
      const cz = currentPositions.current[idx + 2];

      const nx = cx + (tx - cx) * step;
      const ny = cy + (ty - cy) * step;
      const nz = cz + (tz - cz) * step;

      let fx = nx, fy = ny, fz = nz;
      if (targetState === "exploded") {
        const time = state.clock.elapsedTime;
        fx += Math.sin(time + i * 0.1) * 0.03;
        fy += Math.cos(time + i * 0.2) * 0.03;
      }

      currentPositions.current[idx] = nx;
      currentPositions.current[idx + 1] = ny;
      currentPositions.current[idx + 2] = nz;

      dummy.position.set(fx, fy, fz);
      
      // --- Scale Logic based on Color ---
      const r = colors[idx];
      const g = colors[idx+1];
      const b = colors[idx+2];
      
      // Heuristic to detect colors based on RGB values
      const isGold = r > 0.8 && g > 0.7 && b < 0.2;
      const isRed = r > 0.7 && g < 0.2 && b < 0.2;
      const isWhite = r > 0.9 && g > 0.9 && b > 0.9;
      
      let scale = 0.15; // Base Leaf Size
      
      if (isGold) scale = 0.35; // Big Yellow Stars
      else if (isRed) scale = 0.28; // Red Balls
      else if (isWhite) scale = 0.12; // Tiny lights

      dummy.scale.set(scale, scale, scale);
      
      // Rotate leaves randomly
      dummy.rotation.x = i; 
      dummy.rotation.z = i;
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    }
    
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <group ref={groupRef}>
      <instancedMesh ref={meshRef} args={[undefined, undefined, PARTICLE_COUNT]}>
        <octahedronGeometry args={[1, 0]} /> 
        {/* Adjusted Material: Removed global emissive to allow true Green colors */}
        <meshStandardMaterial 
            vertexColors 
            toneMapped={false} 
            roughness={0.4}  // Less shiny, more leafy
            metalness={0.5}  // Slight sheen but not chrome
        />
        <instancedBufferAttribute 
            attach="geometry-attributes-color" 
            args={[colors, 3]} 
        />
      </instancedMesh>
    </group>
  );
};

const Scene = ({ 
    interactionState, 
    handX 
}: { 
    interactionState: InteractionState;
    handX: number;
}) => {
  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 1, 28]} fov={45} />
      <OrbitControls enableZoom={false} enablePan={false} maxPolarAngle={Math.PI/1.8} minPolarAngle={Math.PI/3}/>
      
      {/* Lights */}
      <ambientLight intensity={0.5} /> {/* Increased ambient for green visibility */}
      <spotLight position={[10, 20, 10]} intensity={2} angle={0.5} penumbra={1} color="#FFF8E7" castShadow />
      <pointLight position={[-10, 5, -10]} intensity={1.5} color="#E0F7FA" />
      
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={0.5} />
      <Sparkles count={500} scale={35} size={4} speed={0.3} opacity={0.5} color="#FFF" />
      
      <Snow />

      <Float speed={1.5} rotationIntensity={0.1} floatIntensity={0.3}>
        <ParticleTree targetState={interactionState} rotationTarget={handX} />
      </Float>

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -10, 0]}>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#051005" roughness={0.1} metalness={0.8} />
      </mesh>

      <EffectComposer disableNormalPass>
        {/* Tuned bloom to pick up the Gold/Red but keep Green distinct */}
        <Bloom luminanceThreshold={0.65} mipmapBlur intensity={1.2} radius={0.5} />
        <Noise opacity={0.03} />
        <Vignette eskil={false} offset={0.1} darkness={1.0} />
      </EffectComposer>
    </>
  );
};

const App = () => {
  const [interactionState, setInteractionState] = useState<InteractionState>("tree");
  const [handX, setHandX] = useState(0.5);

  const handleInteractionUpdate = (isOpen: boolean, xPos: number) => {
    setInteractionState(isOpen ? "exploded" : "tree");
    setHandX(xPos); 
  };

  return (
    <div className="relative w-full h-full bg-[#020617]">
      <HandTracker onUpdateInteraction={handleInteractionUpdate} />

      <Canvas 
        gl={{ antialias: false, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.0 }}
        dpr={[1, 2]}
      >
        <Suspense fallback={null}>
            <Scene interactionState={interactionState} handX={handX} />
        </Suspense>
      </Canvas>

      <div className="absolute top-0 left-0 w-full p-8 flex flex-col items-center pointer-events-none select-none z-10">
        <h1 className="text-5xl md:text-7xl font-serif text-transparent bg-clip-text bg-gradient-to-b from-[#D42426] via-[#FFD700] to-[#228B22] drop-shadow-[0_0_15px_rgba(255,215,0,0.3)] font-bold tracking-tight text-center" style={{ fontFamily: 'Times New Roman' }}>
          Merry Christmas
        </h1>
        <div className="h-px w-48 bg-gradient-to-r from-transparent via-red-500 to-transparent mt-3 opacity-60"></div>
      </div>
    </div>
  );
};

const root = createRoot(document.getElementById("root")!);
root.render(<App />);
