#!/usr/bin/env bash
set -euo pipefail

# =====================================
# 1. 초기 설정 및 경로 정의
# =====================================

# 코어 수 설정
CORES=$(($(nproc) / 4))
if [ "$CORES" -lt 1 ]; then CORES=1; fi
echo "[INFO] Using ${CORES} cores."

# 절대 경로 및 GROMACS 설정
ROOT="/home/aaab01/gmx/choi1/"
GMX=$(which gmx)

# PLUMED 경로 설정 (Conda 환경)
export PLUMED_KERNEL="$CONDA_PREFIX/lib/libplumedKernel.so"
echo "[INFO] PLUMED_KERNEL: $PLUMED_KERNEL"

# 실행 디렉토리 생성
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$ROOT/runs/run_$STAMP"

mkdir -p "$RUN_DIR"/{inputs,mdp,top,build,logs,plumed_inputs}

echo "[INFO] GROMACS binary: $GMX"
echo "[INFO] Working directory: $RUN_DIR"

# =====================================
# 2. 파라미터 설정
# =====================================
N_BATH=2000
BOX=13.6
EPS_C=0.3
STEPS_NVT=1000000
STEPS_NVE=4000000
T_INIT=100 

# 틸팅 프로토콜 파라미터
BARRIER_HEIGHT=11.0   # A
WELL_DIST_FIXED=1.0 # b
MAX_TILT=33.8     # C_max

# =====================================
# 3. 파일 복사 및 MDP 설정
# =====================================
cp "$ROOT"/inputs/* "$RUN_DIR"/inputs/
cp "$ROOT"/top/* "$RUN_DIR"/top/
cp "$ROOT"/mdp/* "$RUN_DIR"/mdp/
cd "$RUN_DIR"

echo "[INFO] Configuring MDP files..."
for f in mdp/nvt.mdp mdp/nve.mdp; do
    sed -i "s/^nstxout.*/nstxout = 0/" $f
    sed -i "s/^nstvout.*/nstvout = 0/" $f
    sed -i "s/^nstfout.*/nstfout = 0/" $f
    sed -i "s/^nstxout-compressed.*/nstxout-compressed = 1000/" $f
    sed -i "s/^compressed-x-precision.*/compressed-x-precision = 1000/" $f
done

sed -i "s/^nsteps.*/nsteps = $STEPS_NVT/" mdp/nvt.mdp
sed -i "s/^nsteps.*/nsteps = $STEPS_NVE/" mdp/nve.mdp

# =====================================
# 4. 분자 삽입 (System Setup) - [수정됨: 랜덤 초기화]
# =====================================
echo "[INFO] Setting up system with 50/50 initial state..."

# 50% 확률로 왼쪽(-1.0) 또는 오른쪽(+1.0) 선택
if [ $((RANDOM % 2)) -eq 0 ]; then
    INIT_POS="-1.0"
    echo "[INFO] Initial Bit Position: LEFT (-1.0 nm)"
else
    INIT_POS="1.0"
    echo "[INFO] Initial Bit Position: RIGHT (+1.0 nm)"
fi

# 1. 비트 입자를 선택된 위치(INIT_POS, 0, 0)에 배치
#    -center x y z : 분자의 중심을 해당 좌표로 이동
#    -box : 박스 크기 설정 (여기서 박스를 정의해야 insert-molecules가 인식함)
$GMX editconf -f inputs/bit.gro -o build/bit_pos.gro -center $INIT_POS 0 0 -box $BOX $BOX $BOX

# 2. 나머지 공간에 수조(Bath) 채우기
$GMX insert-molecules -f build/bit_pos.gro -ci inputs/bath.gro -nmol $N_BATH -o build/initial_box.gro

INIT_GRO="build/initial_box.gro"

# =====================================
# 5. 에너지 최소화 (Double EM)
# =====================================
if [ -f "$INIT_GRO" ]; then
    # Phase 1: Soft EM
    echo "[INFO] Phase 1: Soft EM"
    cp mdp/em.mdp mdp/em_soft.mdp
    sed -i "s/^emstep.*/emstep = 0.0001/" mdp/em_soft.mdp
    sed -i "s/^emtol.*/emtol = 1000.0/"   mdp/em_soft.mdp
    sed -i "s/^nsteps.*/nsteps = 100000/"  mdp/em_soft.mdp

    $GMX grompp -f mdp/em_soft.mdp -c "$INIT_GRO" -p top/topol.top -o build/em_soft.tpr -maxwarn 2
    $GMX mdrun -v -deffnm build/em_soft > logs/em_soft.log 2>&1

    # Phase 2: Hard EM
    echo "[INFO] Phase 2: Hard EM"
    cp mdp/em.mdp mdp/em_hard.mdp
    sed -i "s/^emstep.*/emstep = 0.01/"   mdp/em_hard.mdp
    sed -i "s/^emtol.*/emtol = 100.0/"    mdp/em_hard.mdp
    sed -i "s/^nsteps.*/nsteps = 100000/"  mdp/em_hard.mdp

    PREV_EM="build/em_soft.gro"
    if [ ! -f "$PREV_EM" ]; then PREV_EM="$INIT_GRO"; fi

    $GMX grompp -f mdp/em_hard.mdp -c "$PREV_EM" -p top/topol.top -o build/em_hard.tpr -maxwarn 1
    $GMX mdrun -v -deffnm build/em_hard > logs/em_hard.log 2>&1

    INIT_GRO="build/em_hard.gro"
else
    echo "[ERROR] Initial coordinates not found."
    exit 1
fi

# =====================================
# 6. NVT 평형화 (Symmetric Double Well)
# =====================================
echo "[INFO] Running NVT..."
cat << EOF > plumed_inputs/plumed_nvt.dat
p1: POSITION ATOM=1
x1: COMBINE ARG=p1.x COEFFICIENTS=1.0 PERIODIC=NO
dw_pot: CUSTOM ARG=x1 FUNC=$BARRIER_HEIGHT*(x^2-$WELL_DIST_FIXED^2)^2 PERIODIC=NO
bv: BIASVALUE ARG=dw_pot
EOF

$GMX grompp -f mdp/nvt.mdp -c "$INIT_GRO" -p top/topol.top -o build/nvt.tpr -maxwarn 1
$GMX mdrun -ntmpi 1 -ntomp $CORES -gpu_id 3 -nb gpu -update gpu -v -deffnm build/nvt -cpo build/nvt.cpt -plumed plumed_inputs/plumed_nvt.dat > logs/nvt.log 2>&1

INIT_GRO="build/nvt.gro"
INIT_CPT="build/nvt.cpt"

# =====================================
# 7. Full Landauer Cycle (Tilt -> Hold -> Restore)
# =====================================
echo "[INFO] Starting Full Cycle NVE Protocol..."

# 시간 계산
DT_VAL=$(grep -E "^\s*dt\s*=" "mdp/nve.mdp" | sed 's/;.*//' | awk -F '=' '{print $2}' | xargs)
if [ -z "$DT_VAL" ]; then DT_VAL=0.001; fi
NSTEPS_VAL=$(grep -E "^\s*nsteps\s*=" "mdp/nve.mdp" | sed 's/;.*//' | awk -F '=' '{print $2}' | xargs)
if [ -z "$NSTEPS_VAL" ]; then echo "[ERROR] nsteps not found"; exit 1; fi

TOTAL_TIME=$(echo "$NSTEPS_VAL * $DT_VAL" | bc)

# 3단계 시간 분배
T1=$(echo "$TOTAL_TIME * 0.4" | bc) # Tilt End
T2=$(echo "$TOTAL_TIME * 0.6" | bc) # Restore Start
T_RESTORE_DUR=$(echo "$TOTAL_TIME - $T2" | bc)

echo "[INFO] Total: $TOTAL_TIME ps | Tilt: 0-$T1 | Hold: $T1-$T2 | Restore: $T2-$TOTAL_TIME"
echo "[INFO] Max Tilt: $MAX_TILT kJ/mol/nm"

# PLUMED 파일 생성 (Full Cycle Logic)
PLUMED_FILE="plumed_inputs/plumed_nve.dat"

cat << EOF > "$PLUMED_FILE"
UNITS LENGTH=nm TIME=ps ENERGY=kj/mol

p1: POSITION ATOM=1
x1: COMBINE ARG=p1.x COEFFICIENTS=1.0 PERIODIC=NO
t: TIME

# 1. Tilt (0->Max) | 2. Hold (Max) | 3. Restore (Max->0)
tilt_c: MATHEVAL ARG=t FUNC=(1-step(x-$T1))*($MAX_TILT*x/$T1)+step(x-$T1)*(1-step(x-$T2))*$MAX_TILT+step(x-$T2)*$MAX_TILT*(1-(x-$T2)/$T_RESTORE_DUR) PERIODIC=NO

pot: CUSTOM ARG=x1,tilt_c FUNC=$BARRIER_HEIGHT*(x^2-$WELL_DIST_FIXED^2)^2+y*x PERIODIC=NO
bv: BIASVALUE ARG=pot

PRINT ARG=t,x1,tilt_c,pot FILE=logs/colvar_nve.dat STRIDE=1000
EOF

# NVE 실행
$GMX grompp -f mdp/nve.mdp -c "$INIT_GRO" -p top/topol.top -o build/nve.tpr -t "$INIT_CPT" -maxwarn 1
$GMX mdrun -ntmpi 1 -ntomp $CORES -gpu_id 3 -nb gpu -update gpu -v -deffnm build/nve -plumed "$PLUMED_FILE"

echo "[INFO] Simulation Complete."

# =====================================
# 8. 에너지 데이터 추출
# =====================================
echo "[INFO] Extracting energies..."

# (1) NVE
echo -e "Potential\nKinetic-En.\nTotal-Energy\nConserved-En.\nTemperature\n0" | \
$GMX energy -f build/nve.edr -o build/energy_nve.xvg > /dev/null 2>&1

# (2) NVT
if [ -f "build/nvt.edr" ]; then
    echo -e "Potential\nKinetic-En.\nTotal-Energy\nTemperature\n0" | \
    $GMX energy -f build/nvt.edr -o build/energy_nvt.xvg > /dev/null 2>&1
fi

echo "[INFO] All Done."