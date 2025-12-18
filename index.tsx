import React, { useState, useEffect, useRef, Suspense, useMemo } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars, PerspectiveCamera } from "@react-three/drei";
import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// --- Configuration ---
const TREE_HEIGHT = 16;
const TREE_RADIUS = 7;
const EXPLOSION_SPEED = 0.20; 
const RETURN_SPEED = 0.12;
const COUNT_LEAVES = 12000;
const COUNT_RIBBON = 1200;    
const DECO_COUNT = 180; 
const SPIRALS = 12; // 螺旋圈数
const LEAF_COLORS = ["#022c22", "#14532d", "#166534", "#15803d", "#064e3b", "#065f46"];
const RIBBON_COLOR_HEX = "#ffffff"; 

const randomRange = (min: number, max: number) => Math.random() * (max - min) + min;
const getDistance = (p1: {x:number, y:number}, p2: {x:number, y:number}) => Math.hypot(p1.x - p2.x, p1.y - p2.y);
const isFingerExtended = (landmarks: any[], tipIdx: number, pipIdx: number) => 
    getDistance(landmarks[0], landmarks[tipIdx]) > getDistance(landmarks[0], landmarks[pipIdx]);

const HandTracker = ({ onUpdateInteraction }: { onUpdateInteraction: (factor: number, rotation: number | null) => void }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [status, setStatus] = useState("Loading AI...");
  const [gestureText, setGestureText] = useState("Waiting...");
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const reqIdRef = useRef<number>(0);

  useEffect(() => {
    let isMounted = true;
    const init = async () => {
        try {
            const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm");
            if (!isMounted) return;
            const landmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, delegate: "GPU" },
                runningMode: "VIDEO", numHands: 1
            });
            if (isMounted) { landmarkerRef.current = landmarker; setModelLoaded(true); setStatus("Ready"); }
        } catch (e) { if (isMounted) setStatus("Error: AI Model Failed"); }
    };
    init();
    return () => { isMounted = false; };
  }, []);

  const drawSkeleton = (ctx: CanvasRenderingContext2D, landmarks: any[], width: number, height: number, isOpen: boolean) => {
      ctx.clearRect(0, 0, width, height);
      ctx.lineWidth = 2; ctx.strokeStyle = isOpen ? "#34d399" : "#f87171"; ctx.fillStyle = isOpen ? "#34d399" : "#f87171";
      const connections = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]];
      connections.forEach(row => {
          ctx.beginPath();
          row.forEach((idx, i) => {
              const x = (1 - landmarks[idx].x) * width; const y = landmarks[idx].y * height;
              if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          });
          ctx.stroke();
      });
      landmarks.forEach((p: any) => {
          const x = (1 - p.x) * width; const y = p.y * height;
          ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill();
      });
  };

  const predict = () => {
    const video = videoRef.current; const canvas = canvasRef.current;
    if (video && landmarkerRef.current && canvas && video.readyState >= 2) {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
            try {
                if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
                }
                const result = landmarkerRef.current.detectForVideo(video, performance.now());
                const ctx = canvas.getContext('2d');
                if (ctx && result.landmarks && result.landmarks.length > 0) {
                    const hand = result.landmarks[0];
                    const extendedFingers = [isFingerExtended(hand, 8, 6), isFingerExtended(hand, 12, 10), isFingerExtended(hand, 16, 14), isFingerExtended(hand, 20, 18)].filter(Boolean).length;
                    const totalExtended = extendedFingers + (isFingerExtended(hand, 4, 2) ? 1 : 0);
                    let targetFactor = (totalExtended >= 4) ? 1 : 0;
                    let text = (totalExtended >= 4) ? "Open Palm (Explode)" : "Fist (Rotate)";
                    let isOpen = totalExtended >= 4;
                    const rotation = (0.5 - hand[0].x) * 5;
                    onUpdateInteraction(targetFactor, rotation); setGestureText(text);
                    drawSkeleton(ctx, hand, canvas.width, canvas.height, isOpen);
                } else if (ctx) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height); onUpdateInteraction(0, null); setGestureText("No Hand Detected");
                }
            } catch (err) {}
        }
    }
    reqIdRef.current = requestAnimationFrame(predict);
  };

  const startCamera = async () => {
    if (!landmarkerRef.current) return;
    setIsCameraActive(true); setStatus("Starting Camera...");
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30 } });
        if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (err) { setStatus("Error: Camera Denied."); setIsCameraActive(false); }
  };

  return (
    <div className="absolute bottom-4 left-4 z-50 p-4 bg-slate-900/90 backdrop-blur border border-emerald-500/30 rounded-lg max-w-sm shadow-2xl transition-all duration-300">
        {!isCameraActive ? (
            <div className="text-center">
                <p className="text-emerald-100 mb-3 font-sans text-sm font-medium tracking-wide">{modelLoaded ? "Use gestures to control the tree!" : status}</p>
                <button onClick={startCamera} disabled={!modelLoaded} className={`px-6 py-2 font-bold rounded shadow-lg transition-all font-sans text-sm text-white ${modelLoaded ? "bg-gradient-to-r from-emerald-700 to-emerald-500 hover:from-emerald-600 hover:to-emerald-400" : "bg-slate-700 opacity-50"}`}>
                    {modelLoaded ? "Start Magic" : "Wait..."}
                </button>
            </div>
        ) : (
             <div className="flex items-center gap-4">
                 <div className="relative w-32 h-24 bg-black rounded-lg overflow-hidden border-2 border-emerald-500/50">
                    <video ref={videoRef} autoPlay playsInline muted onCanPlay={() => { if (videoRef.current) { videoRef.current.play(); setStatus("Active"); if (!reqIdRef.current) reqIdRef.current = requestAnimationFrame(predict); } }} className="absolute w-full h-full object-cover transform scale-x-[-1] opacity-40" />
                    <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full object-cover pointer-events-none" />
                 </div>
                 <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className={`w-2 h-2 rounded-full animate-pulse ${gestureText.includes("Open") ? "bg-emerald-400" : "bg-red-400"}`}></span>
                        <p className="text-sm font-bold font-sans text-slate-300">{gestureText}</p>
                    </div>
                    <p className="text-[10px] text-slate-400 font-sans leading-tight mt-1"><strong className="text-emerald-400">Open Palm</strong> to Explode<br/><strong className="text-red-400">Fist + Move</strong> to Rotate</p>
                 </div>
             </div>
        )}
    </div>
  );
};

const TreeParticles = ({ targetFactor }: { targetFactor: number }) => {
  const meshRef = useRef<THREE.Points>(null);
  const { initialPositions, explodedPositions, colors, sizes } = useMemo(() => {
    const total = COUNT_LEAVES + COUNT_RIBBON;
    const posInit = new Float32Array(total * 3), posExp = new Float32Array(total * 3), col = new Float32Array(total * 3), siz = new Float32Array(total);
    let idx = 0;
    for (let i = 0; i < COUNT_LEAVES; i++) {
        const y = Math.random() * TREE_HEIGHT; const radiusAtY = TREE_RADIUS * (1 - y / TREE_HEIGHT);
        const r = Math.pow(Math.random(), 0.3) * radiusAtY; const theta = Math.random() * Math.PI * 2;
        posInit[idx*3] = r * Math.cos(theta); posInit[idx*3+1] = y - TREE_HEIGHT/2; posInit[idx*3+2] = r * Math.sin(theta);
        const expR = randomRange(20, 50), expTheta = Math.random() * Math.PI * 2, expPhi = Math.acos(2 * Math.random() - 1);
        posExp[idx*3] = expR * Math.sin(expPhi) * Math.cos(expTheta); posExp[idx*3+1] = expR * Math.sin(expPhi) * Math.sin(expTheta); posExp[idx*3+2] = expR * Math.cos(expPhi);
        const c = new THREE.Color(LEAF_COLORS[Math.floor(Math.random() * LEAF_COLORS.length)]);
        col[idx*3] = c.r; col[idx*3+1] = c.g; col[idx*3+2] = c.b; siz[idx] = randomRange(0.25, 0.45); idx++;
    }
    const ribbonColor = new THREE.Color(RIBBON_COLOR_HEX);
    for (let i = 0; i < COUNT_RIBBON; i++) {
        const t = i / COUNT_RIBBON, y = t * TREE_HEIGHT, radiusAtY = (TREE_RADIUS * (1 - y / TREE_HEIGHT)) + 0.3, theta = t * Math.PI * 2 * SPIRALS;
        posInit[idx*3] = radiusAtY * Math.cos(theta); posInit[idx*3+1] = y - TREE_HEIGHT/2; posInit[idx*3+2] = radiusAtY * Math.sin(theta);
        const expR = randomRange(25, 45); posExp[idx*3] = expR * Math.cos(theta); posExp[idx*3+1] = (t - 0.5) * 5; posExp[idx*3+2] = expR * Math.sin(theta);
        col[idx*3] = ribbonColor.r * 3; col[idx*3+1] = ribbonColor.g * 3; col[idx*3+2] = ribbonColor.b * 3; siz[idx] = 0.5; idx++;
    }
    return { initialPositions: posInit, explodedPositions: posExp, colors: col, sizes: siz };
  }, []);
  const currentPositions = useMemo(() => new Float32Array(initialPositions), [initialPositions]);
  useFrame(() => {
    if (!meshRef.current) return;
    const speed = targetFactor > 0.1 ? EXPLOSION_SPEED : RETURN_SPEED;
    for (let i = 0; i < currentPositions.length / 3; i++) {
        const ix = i * 3, iy = i * 3 + 1, iz = i * 3 + 2;
        const tx = initialPositions[ix] + (explodedPositions[ix] - initialPositions[ix]) * targetFactor;
        const ty = initialPositions[iy] + (explodedPositions[iy] - initialPositions[iy]) * targetFactor;
        const tz = initialPositions[iz] + (explodedPositions[iz] - initialPositions[iz]) * targetFactor;
        currentPositions[ix] += (tx - currentPositions[ix]) * speed; currentPositions[iy] += (ty - currentPositions[iy]) * speed; currentPositions[iz] += (tz - currentPositions[iz]) * speed;
    }
    meshRef.current.geometry.attributes.position.needsUpdate = true;
  });
  return (
    <points ref={meshRef}>
        <bufferGeometry>
            <bufferAttribute attach="attributes-position" count={currentPositions.length / 3} array={currentPositions} itemSize={3} />
            <bufferAttribute attach="attributes-color" count={colors.length / 3} array={colors} itemSize={3} />
            <bufferAttribute attach="attributes-size" count={sizes.length} array={sizes} itemSize={1} />
        </bufferGeometry>
        <pointsMaterial vertexColors size={0.5} sizeAttenuation={true} transparent opacity={0.9} blending={THREE.NormalBlending} depthWrite={false} />
    </points>
  );
};

const GiftDeco = () => (<group scale={0.4}><mesh><boxGeometry args={[1,1,1]}/><meshStandardMaterial color="#dc2626"/></mesh><mesh scale={[1.05,1.05,0.2]}><boxGeometry/><meshStandardMaterial color="#fbbf24"/></mesh><mesh scale={[0.2,1.05,1.05]}><boxGeometry/><meshStandardMaterial color="#fbbf24"/></mesh></group>);
const HatDeco = () => (<group scale={0.4} rotation={[0.2,0,0]}><mesh position={[0,0.5,0]}><coneGeometry args={[0.6,1.2,32]}/><meshStandardMaterial color="#dc2626"/></mesh><mesh position={[0,-0.1,0]} rotation={[Math.PI/2,0,0]}><torusGeometry args={[0.6,0.2]}/><meshStandardMaterial color="#f8fafc"/></mesh></group>);
const CaneDeco = () => (<group scale={0.4}><mesh position={[0,0.5,0]}><torusGeometry args={[0.3,0.1,8,16,Math.PI]}/><meshStandardMaterial color="#ffffff"/></mesh><mesh position={[0.3,0,0]}><cylinderGeometry args={[0.1,0.1,1]}/><meshStandardMaterial color="#ffffff"/></mesh></group>);
const BaubleDeco = () => (<group scale={0.4}><mesh><sphereGeometry args={[0.8,24,24]}/><meshStandardMaterial color="#991b1b" metalness={0.7}/></mesh></group>);

const ComplexDecorationSystem = ({ targetFactor }: { targetFactor: number }) => {
    const items = useMemo(() => {
        const _items = []; const types = ['gift', 'hat', 'cane', 'bauble'];
        for (let i = 0; i < DECO_COUNT; i++) {
            const y = Math.random() * TREE_HEIGHT; const radiusAtY = TREE_RADIUS * (1 - y / TREE_HEIGHT);
            const r = (0.7 + 0.3 * Math.random()) * radiusAtY; const theta = Math.random() * Math.PI * 2;
            const startPos = new THREE.Vector3(r * Math.cos(theta), y - TREE_HEIGHT / 2, r * Math.sin(theta));
            const expR = randomRange(15, 60), expTheta = Math.random() * Math.PI * 2, expPhi = Math.acos(2 * Math.random() - 1);
            const endPos = new THREE.Vector3(expR * Math.sin(expPhi) * Math.cos(expTheta), expR * Math.sin(expPhi) * Math.sin(expTheta), expR * Math.cos(expPhi));
            _items.push({ id: i, startPos, endPos, type: types[i % 4], rot: [Math.random()*Math.PI, Math.random()*Math.PI, Math.random()*Math.PI] });
        }
        return _items;
    }, []);
    const groupRef = useRef<THREE.Group>(null); const curF = useRef(0);
    useFrame((state) => {
        if (!groupRef.current) return;
        curF.current += (targetFactor - curF.current) * (targetFactor > 0.1 ? EXPLOSION_SPEED : RETURN_SPEED);
        groupRef.current.children.forEach((child, i) => {
            const it = items[i]; child.position.lerpVectors(it.startPos, it.endPos, curF.current);
            child.rotation.set(it.rot[0] + state.clock.elapsedTime, it.rot[1] + state.clock.elapsedTime, it.rot[2]);
        });
    });
    return (<group ref={groupRef}>{items.map(it => (<group key={it.id}>{it.type==='gift' && <GiftDeco/>}{it.type==='hat' && <HatDeco/>}{it.type==='cane' && <CaneDeco/>}{it.type==='bauble' && <BaubleDeco/>}</group>))}</group>);
};

const TopStar = ({ targetFactor }: { targetFactor: number }) => {
    const ref = useRef<THREE.Group>(null);
    useFrame((state) => {
        if (!ref.current) return;
        const targetY = (TREE_HEIGHT / 2) + 0.5 + (targetFactor * 10) + Math.sin(state.clock.elapsedTime * 2) * 0.2;
        ref.current.position.y += (targetY - ref.current.position.y) * 0.1; ref.current.rotation.y = state.clock.elapsedTime * 0.5;
    });
    return (
        <group ref={ref}>
             <mesh><icosahedronGeometry args={[0.4, 0]} /><meshBasicMaterial color="#fef08a" toneMapped={false} /></mesh>
             <mesh scale={1.2}><icosahedronGeometry args={[0.45, 1]} /><meshBasicMaterial color="#fcd34d" transparent opacity={0.4} toneMapped={false} /></mesh>
             <pointLight intensity={1.5} color="#fef08a" distance={8} decay={2} />
        </group>
    );
};

const Scene = ({ interactionFactor, treeRotation }: { interactionFactor: number, treeRotation: number | null }) => {
    const groupRef = useRef<THREE.Group>(null);
    useFrame((_, delta) => {
        if (!groupRef.current) return;
        if (treeRotation !== null) groupRef.current.rotation.y = THREE.MathUtils.lerp(groupRef.current.rotation.y, treeRotation, 0.1);
        else groupRef.current.rotation.y += delta * 0.2;
    });
    return (
        <>
            <OrbitControls enablePan={false} minPolarAngle={Math.PI / 2.5} maxPolarAngle={Math.PI / 1.8} enableRotate={treeRotation === null} maxDistance={45} minDistance={15} />
            <PerspectiveCamera makeDefault position={[0, 4, 32]} fov={50} />
            <ambientLight intensity={1.5} color="#fff1f2" />
            <spotLight position={[20, 30, 10]} intensity={2.5} />
            <color attach="background" args={['#020617']} />
            <Stars radius={100} depth={50} count={3000} factor={4} />
            <group ref={groupRef}>
                 <TopStar targetFactor={interactionFactor} />
                 <TreeParticles targetFactor={interactionFactor} />
                 <ComplexDecorationSystem targetFactor={interactionFactor} />
            </group>
            <EffectComposer disableNormalPass={false}>
                <Bloom luminanceThreshold={0.7} mipmapBlur intensity={1.5} radius={0.5} />
                <Vignette eskil={false} offset={0.1} darkness={0.6} />
            </EffectComposer>
        </>
    );
};

const App = () => {
  const [interactionState, setInteractionState] = useState<{factor: number, rotation: number | null}>({ factor: 0, rotation: null });
  return (
    <>
        <div className="absolute top-0 left-0 w-full p-8 z-10 pointer-events-none select-none text-center">
            <h1 className="text-5xl md:text-7xl font-bold gold-text tracking-wider opacity-90 drop-shadow-lg">Merry Christmas</h1>
            <p className="text-emerald-200/70 mt-2 font-sans text-sm tracking-[0.3em] uppercase">Wish u a nice day</p>
        </div>
        <HandTracker onUpdateInteraction={(factor, rotation) => setInteractionState({ factor, rotation })} />
        <Canvas gl={{ antialias: false, toneMapping: THREE.ReinhardToneMapping, toneMappingExposure: 1.5 }} dpr={[1, 2]}>
            <Suspense fallback={null}><Scene interactionFactor={interactionState.factor} treeRotation={interactionState.rotation} /></Suspense>
        </Canvas>
    </>
  );
};

createRoot(document.getElementById("root")!).render(<App />);