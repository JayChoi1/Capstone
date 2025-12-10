#!/usr/bin/env python3
"""
Landauer Bit Simulation Comprehensive Analysis Tool
[작성일] 2025-12-04
[기능]
 1. NVT 평형화 검증 (온도 안정성)
 2. NVE 연속 틸팅 궤적 및 프로토콜 시각화
 3. Landauer Work vs Total Energy 검증 (열 발생량 계산)
 4. 시스템(욕조) 온도 상승 분석
 5. 위상 공간(Phase Space) 궤적 분석 (New)
 6. 열 발생률(Heat Flux) 분석 (New)
 7. 퍼텐셜 애니메이션 생성
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import pandas as pd

# 폰트 설정
try:
    import matplotlib
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass

# ===== 상수 정의 =====
BARRIER_HEIGHT = 11.0   # A
WELL_DIST_FIXED = 1.0   # b
MAX_TILT = 33.8       # C_max

def get_latest_run_dir(root_dir):
    """최근 Run 디렉토리 자동 탐색"""
    # 1. 상위 폴더 검색
    search_path = os.path.join(root_dir, '..', 'runs', 'run_*')
    runs = sorted(glob.glob(search_path))
    
    # 2. 현재 폴더 하위 검색
    if not runs:
        search_path = os.path.join(root_dir, 'runs', 'run_*')
        runs = sorted(glob.glob(search_path))
        
    # 3. 현재 폴더가 run 폴더인 경우
    if not runs and os.path.exists(os.path.join(root_dir, 'build')):
        return root_dir

    return runs[-1] if runs else None

def read_xvg(filepath):
    """GROMACS .xvg 파일 안전 로더 (주석 제거 및 강제 파싱)"""
    if not os.path.exists(filepath):
        print(f"[WARN] 파일을 찾을 수 없음: {os.path.basename(filepath)}")
        return None
    
    clean_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith(('#', '@')):
                continue
            clean_lines.append(line)
            
    from io import StringIO
    tmp_data = StringIO("".join(clean_lines))
    
    # 일반적인 에너지 파일 컬럼 (Time, Pot, Kin, Tot, Temp, etc...)
    # NVT와 NVE의 컬럼 수가 다를 수 있으므로 이름 없이 로드 후 매핑
    try:
        df = pd.read_csv(tmp_data, sep=r'\s+', header=None, engine='python')
        return df
    except Exception as e:
        print(f"[ERROR] XVG 파싱 실패: {e}")
        return None

def read_colvar(filepath):
    """PLUMED colvar 파일 로더"""
    if not os.path.exists(filepath):
        print(f"[ERROR] 파일을 찾을 수 없음: {filepath}")
        return None
    
    col_names = None
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#! FIELDS'):
                col_names = line.split()[2:]
                break
    try:
        df = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None, names=col_names, engine='python')
        return df
    except Exception as e:
        print(f"[ERROR] colvar 파싱 실패: {e}")
        return None

# =========================================================
# [분석 모듈 1] NVT 평형화 분석 (추가됨)
# =========================================================
def analyze_nvt(run_dir, graph_dir):
    print("[INFO] NVT 평형화 데이터 분석 중...")
    nvt_file = os.path.join(run_dir, 'build', 'energy_nvt.xvg')
    
    df = read_xvg(nvt_file)
    if df is None:
        return

    # NVT 파일 컬럼 추정 (보통 Time, Pot, Kin, Tot, Temp, Pres, Vol...)
    # 사용자가 스크립트에서 "3 4 5 7 0" (Pot, Kin, Tot, Temp) 순으로 뽑았다고 가정
    # 컬럼 인덱스: 0=Time, 1=Pot, 2=Kin, 3=Tot, 4=Temp (총 5개일 경우)
    
    if df.shape[1] >= 5:
        time = df.iloc[:, 0]
        temp = df.iloc[:, 4]
        
        plt.figure(figsize=(10, 5))
        plt.plot(time, temp, 'k-', alpha=0.5, lw=1)
        plt.axhline(100, color='r', ls='--', label='Target (100 K)')
        
        # 통계
        mean_T = temp.mean()
        std_T = temp.std()
        
        plt.title(f"NVT Equilibration Check\nMean: {mean_T:.2f} K, Std: {std_T:.2f} K")
        plt.xlabel("Time (ps)")
        plt.ylabel("Temperature (K)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(graph_dir, '00_nvt_equilibration.png'))
        plt.close()
        print(f"[RESULT] NVT Mean Temp: {mean_T:.2f} K")
    else:
        print("[WARN] NVT 데이터 컬럼 수 부족으로 그래프 생략")

# =========================================================
# [분석 모듈 2] 위상 공간 & 열 발생률 (추가됨)
# =========================================================
def analyze_phase_space_and_heat_flux(df_col, drift_series, time_series, graph_dir):
    print("[INFO] 위상 공간 및 열 발생률 분석 중...")
    
    # 1. 위상 공간 (Phase Space): x vs v
    # 속도 추정 (v = dx/dt)
    dt = df_col['time'].diff()
    dx = df_col['x1'].diff()
    v = dx / dt
    
    plt.figure(figsize=(8, 8))
    # 시간에 따라 색상 변화 (초기:파랑 -> 후기:빨강)
    plt.scatter(df_col['x1'], v, c=df_col['time'], cmap='turbo', s=1, alpha=0.5)
    plt.colorbar(label='Time (ps)')
    
    plt.title("Phase Space Trajectory (x vs v)")
    plt.xlabel("Position x (nm)")
    plt.ylabel("Velocity v (nm/ps)")
    plt.grid(True)
    plt.xlim(-2.0, 2.0)
    # v 범위는 데이터에 맞게 자동 조정되지만, 너무 튀는 값 제외
    v_clean = v[np.abs(v) < v.std()*5]
    if not v_clean.empty:
        plt.ylim(v_clean.min(), v_clean.max())
        
    plt.savefig(os.path.join(graph_dir, '07_phase_space.png'))
    plt.close()
    
    # 2. 열 발생률 (Heat Flux = d(Drift)/dt)
    # drift_series는 Energy 데이터 기준이므로 시간축이 다를 수 있음
    drift_rate = np.gradient(drift_series, time_series)
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, drift_rate, 'r-', lw=1)
    plt.axhline(0, color='k', ls='--')
    plt.title("Heat Dissipation Rate (dQ/dt)")
    plt.xlabel("Time (ps)")
    plt.ylabel("Heat Flux (kJ/mol/ps)")
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, '08_heat_flux.png'))
    plt.close()

# =========================================================
# [분석 모듈 3] 온도 상세 분석
# =========================================================
def analyze_temperatures(df_eng, graph_dir):
    # System Temperature (Bath)
    # df_eng 컬럼: 0=Time, 1=Pot, 2=Kin, 3=Tot, 4=Temp
    if df_eng.shape[1] < 5: return

    time = df_eng.iloc[:, 0]
    temp = df_eng.iloc[:, 4]

    plt.figure(figsize=(10, 6))
    plt.plot(time, temp, 'k-', alpha=0.2, label='Raw Data')
    
    # 이동 평균
    window = max(1, len(df_eng) // 50)
    temp_smooth = temp.rolling(window=window, center=True).mean()
    plt.plot(time, temp_smooth, 'r-', lw=2, label=f'Moving Avg')
    
    delta_T = temp_smooth.iloc[-window] - temp_smooth.iloc[window] if len(temp_smooth) > window else 0
    
    plt.title(f"System (Bath) Temperature Rise\nDelta T approx {delta_T:.2f} K")
    plt.xlabel("Time (ps)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, '05_system_temperature.png'))
    plt.close()

# =========================================================
# [메인 분석 컨트롤러]
# =========================================================
def analyze_full_simulation(run_dir):
    print(f"\n========== 종합 분석 시작: {os.path.basename(run_dir)} ==========")
    graph_dir = os.path.join(run_dir, 'graphs_full')
    os.makedirs(graph_dir, exist_ok=True)

    # 1. NVT 분석 실행
    analyze_nvt(run_dir, graph_dir)

    # 2. NVE 데이터 로드
    xvg_file = os.path.join(run_dir, 'build', 'energy_nve_continuous.xvg')
    col_file = os.path.join(run_dir, 'logs', 'colvar_nve_continuous.dat')

    # 파일명 호환성 체크
    if not os.path.exists(xvg_file): xvg_file = os.path.join(run_dir, 'build', 'energy_nve.xvg')
    if not os.path.exists(col_file): col_file = os.path.join(run_dir, 'logs', 'colvar_nve.dat')

    df_eng = read_xvg(xvg_file)
    df_col = read_colvar(col_file)

    if df_eng is None or df_col is None:
        print("[CRITICAL] NVE 데이터 로드 실패. 종료.")
        return

    # df_eng 컬럼 매핑 (Time, Pot, Kin, Tot, Temp)
    # read_xvg는 header=None으로 읽으므로 컬럼 번호로 접근
    eng_time = df_eng.iloc[:, 0]
    eng_tot = df_eng.iloc[:, 3]

    # 3. 궤적 & 프로토콜 시각화
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df_col['time'], df_col['x1'], 'k-', lw=1)
    plt.ylabel('Position (nm)')
    plt.title('Particle Trajectory')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df_col['time'], df_col['tilt_c'], 'r-')
    plt.ylabel('Tilt C')
    plt.xlabel('Time (ps)')
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, '01_trajectory.png'))
    plt.close()

    # 4. Landauer Work & Heat 분석
    dC = df_col['tilt_c'].diff().fillna(0)
    dW = df_col['x1'] * dC
    work_cum = dW.cumsum()
    work_interp = np.interp(eng_time, df_col['time'], work_cum)
    
    delta_E = eng_tot - eng_tot.iloc[0]
    drift = delta_E - work_interp  # Dissipated Heat

    plt.figure(figsize=(10, 6))
    plt.plot(eng_time, delta_E, 'k-', lw=2, label='Delta Total Energy')
    plt.plot(eng_time, work_interp, 'r--', lw=2, label='External Work')
    plt.fill_between(eng_time, delta_E, work_interp, color='gray', alpha=0.2, label='Dissipated Heat (Drift)')
    plt.title(f"Landauer Verification (Heat: {drift.iloc[-1]:.2f} kJ/mol)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, '02_verification.png'))
    plt.close()

    # 5. 추가 분석 모듈 실행
    analyze_temperatures(df_eng, graph_dir)
    analyze_phase_space_and_heat_flux(df_col, drift, eng_time, graph_dir)

    # 6. 애니메이션 생성
    save_continuous_animation(df_col, os.path.join(graph_dir, '04_potential_movie.mp4'))
    print(f"\n[완료] 모든 결과가 저장됨: {graph_dir}")

def save_continuous_animation(df_colvar, out_path):
    print("[INFO] 애니메이션 생성 중...")
    n_frames = 200
    if len(df_colvar) > n_frames:
        idx = np.linspace(0, len(df_colvar)-1, n_frames).astype(int)
        df_sub = df_colvar.iloc[idx]
    else:
        df_sub = df_colvar

    fig, ax = plt.subplots(figsize=(8, 6))
    x_grid = np.linspace(-2.5, 2.5, 300)
    line_pot, = ax.plot([], [], 'b-', lw=2)
    point_part, = ax.plot([], [], 'ro', ms=10)
    txt = ax.text(0.05, 0.95, '', transform=ax.transAxes)

    ax.set_xlim(-2.5, 2.5)
    u_min = -BARRIER_HEIGHT * (WELL_DIST_FIXED**4)
    ax.set_ylim(u_min*1.5, u_min*-1.5 + 200)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("U (kJ/mol)")
    ax.grid(True, alpha=0.3)

    def update(frame):
        row = df_sub.iloc[frame]
        t = row['time']
        x = row['x1']
        C = row['tilt_c'] if 'tilt_c' in row else MAX_TILT * (t/df_sub.iloc[-1]['time'])
        
        U = BARRIER_HEIGHT * (x_grid**2 - WELL_DIST_FIXED**2)**2 + C * x_grid
        line_pot.set_data(x_grid, U)
        point_part.set_data([x], [BARRIER_HEIGHT*(x**2-WELL_DIST_FIXED**2)**2 + C*x])
        txt.set_text(f"t={t:.0f}ps, C={C:.1f}")
        return line_pot, point_part, txt

    ani = animation.FuncAnimation(fig, update, frames=len(df_sub), blit=True)
    try:
        ani.save(out_path, fps=30, extra_args=['-vcodec', 'libx264'])
    except:
        try: ani.save(out_path.replace('.mp4','.gif'), fps=30, writer='pillow')
        except: pass
    plt.close()

if __name__ == "__main__":
    current_root = os.getcwd()
    latest_run = get_latest_run_dir(current_root)
    if latest_run:
        analyze_full_simulation(latest_run)
    else:
        print("[ERROR] Run 디렉토리를 찾을 수 없습니다.")