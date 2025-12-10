#!/usr/bin/env python3
"""
Jarzynski Equality Analysis Script (Modified)
[수정 내용]
 1. 기본적으로 모든 샘플(All)을 분석 (옵션 -n 0)
 2. -n 옵션으로 특정 개수 지정 가능
 3. 파일명 포맷 변경: YYMMDD_HHMM_{SampleCount}.png
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from scipy.special import logsumexp
from scipy.stats import norm
from datetime import datetime  # 날짜/시간 포맷을 위해 추가

# 폰트 설정
try:
    import matplotlib
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass

# ===== 기본 설정 (Default Values) =====
DEFAULT_N_RUNS = 0         # 0이면 모든 파일 분석 (Default: All)
DEFAULT_TEMP_K = 100.0     # 기본 온도
KB = 0.008314462           # 볼츠만 상수 (kJ/mol/K)
OUTPUT_DIR = "JE_analysis" # 결과 저장 폴더명

def get_n_latest_runs(root_dir, n):
    """
    최근 생성된 순서대로 n개의 run 디렉토리를 반환.
    n <= 0 이면 모든 디렉토리를 반환.
    """
    # 상위 폴더의 runs 검색
    search_path = os.path.join(root_dir, '..', 'runs', 'run_*')
    runs = sorted(glob.glob(search_path), key=os.path.getmtime, reverse=True)
    
    # 없으면 현재 폴더의 runs 검색
    if not runs:
        search_path = os.path.join(root_dir, 'runs', 'run_*')
        runs = sorted(glob.glob(search_path), key=os.path.getmtime, reverse=True)
    
    # 그래도 없으면 build 폴더 확인 (단일 런)
    if not runs:
        if os.path.exists(os.path.join(root_dir, 'build')):
             return [root_dir]
        print("[ERROR] 'runs' 디렉토리를 찾을 수 없습니다.")
        return []
    
    # n이 양수일 때만 슬라이싱, 0 이하(기본값)면 전체 반환
    if n > 0:
        return runs[:n]
    
    return runs

def calculate_work_from_colvar(colvar_path):
    """colvar 파일에서 Work 계산"""
    try:
        col_names = None
        with open(colvar_path, 'r') as f:
            for line in f:
                if line.startswith('#! FIELDS'):
                    col_names = line.split()[2:]
                    break
        
        if not col_names: return None

        df = pd.read_csv(colvar_path, sep=r'\s+', comment='#', header=None, names=col_names, engine='python')
        
        dC = df['tilt_c'].diff().fillna(0)
        dW = df['x1'] * dC
        total_work = dW.sum()
        
        return total_work
    except Exception as e:
        print(f"[WARN] {os.path.basename(colvar_path)} 오류: {e}")
        return None

def parse_arguments():
    """터미널 인자 파싱"""
    parser = argparse.ArgumentParser(description="Analyze Jarzynski Equality from simulation runs.")
    parser.add_argument('-n', '--num', type=int, default=DEFAULT_N_RUNS,
                        help=f'Number of recent runs to analyze (Default: 0 -> ALL runs)')
    parser.add_argument('-t', '--temp', type=float, default=DEFAULT_TEMP_K,
                        help=f'Simulation Temperature in Kelvin (Default: {DEFAULT_TEMP_K})')
    return parser.parse_args()

def main():
    # 1. 인자 처리
    args = parse_arguments()
    n_runs_arg = args.num # 사용자 입력 값 (0이면 전체)
    temp_k = args.temp
    beta = 1.0 / (KB * temp_k)
    
    # Landauer 이론 값 계산
    landauer_limit = KB * temp_k * np.log(2)

    # 2. 결과 저장 폴더 확인
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"[INFO] '{OUTPUT_DIR}' 폴더가 생성되었습니다.")
        except OSError:
            print(f"[ERROR] 결과 저장 폴더 '{OUTPUT_DIR}' 생성 실패.")
            return

    current_root = os.getcwd()
    # 0이면 전체, 양수면 개수만큼 가져오기
    target_runs = get_n_latest_runs(current_root, n_runs_arg)
    
    print(f"========== Jarzynski Analysis (Runs Found: {len(target_runs)}, T={temp_k}K) ==========")
    print(f"[THEORY] Landauer Limit (kB T ln 2) = {landauer_limit:.4f} kJ/mol")

    if not target_runs:
        print("[ERROR] 분석할 시뮬레이션 폴더를 찾지 못했습니다.")
        return

    works = []
    run_ids = []

    # 3. Work 추출
    print(f"[INFO] {len(target_runs)}개 시뮬레이션 데이터 로드 중...")
    for run in target_runs:
        col_file = os.path.join(run, 'logs', 'colvar_nve_continuous.dat')
        if not os.path.exists(col_file):
             col_file = os.path.join(run, 'logs', 'colvar_nve.dat')
        
        run_name = os.path.basename(run)
        
        if os.path.exists(col_file):
            w = calculate_work_from_colvar(col_file)
            if w is not None:
                works.append(w)
                run_ids.append(run_name)
                print(f"  - {run_name}: Work = {w:.4f} kJ/mol")

    works = np.array(works)
    n_samples = len(works)

    if n_samples < 1:
        print("[ERROR] 유효한 Work 데이터가 없습니다.")
        return

    # 4. Jarzynski Equality 계산
    neg_beta_w = -beta * works
    log_avg_exp = logsumexp(neg_beta_w) - np.log(n_samples)
    delta_F = - (1.0 / beta) * log_avg_exp
    
    avg_work = np.mean(works)
    std_work = np.std(works)
    dissipated_work = avg_work - delta_F
    
    # 수렴성 데이터
    running_df = []
    for i in range(1, n_samples + 1):
        subset = works[:i]
        subset_nbw = -beta * subset
        val = - (1.0 / beta) * (logsumexp(subset_nbw) - np.log(i))
        running_df.append(val)

    print("\n" + "="*50)
    print(f" [최종 결과] 사용된 샘플 수: {n_samples}")
    print(f" Jarzynski ΔF:     {delta_F:.4f} kJ/mol")
    print(f" Landauer Limit:   {landauer_limit:.4f} kJ/mol")
    print("="*50 + "\n")

    # 5. 시각화
    fig = plt.figure(figsize=(14, 13)) 
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 0.9, 0.6])

    # (1) 히스토그램
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(works, bins=max(3, int(n_samples/2)), density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Work Samples')
    
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    if std_work > 0:
        p = norm.pdf(x, avg_work, std_work)
        ax1.plot(x, p, 'k--', linewidth=2, label='Gaussian Fit')

    ax1.axvline(avg_work, color='blue', linestyle='-', linewidth=2, label=r'Mean Work $\langle W \rangle$')
    ax1.axvline(delta_F, color='red', linestyle='-', linewidth=2, label=r'Jarzynski $\Delta F$')
    ax1.set_title(f'Work Distribution (N={n_samples})')
    ax1.set_xlabel('Work (kJ/mol)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (2) 수렴도
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, n_samples + 1), running_df, 'r-o', label='Jarzynski Estimate')
    ax2.axhline(delta_F, color='red', linestyle='--', alpha=0.5, label='Final Value')
    ax2.set_title(r'Convergence of $\Delta F$ Estimate')
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel(r'Free Energy $\Delta F$ (kJ/mol)')
    ax2.grid(True, alpha=0.3)

    # (3) 수식 텍스트
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    eq_text = (
        "Theory & Calculation:" + "\n"
        r"$\bullet\ T = " + f"{temp_k:.1f}" + r"\ K$" + "\n"
        r"$\bullet\ Landauer\ Limit\ (k_B T \ln 2) = " + f"{landauer_limit:.4f}" + r"\ kJ/mol$" + "\n\n"
        r"$\bullet\ \langle W \rangle = " + f"{avg_work:.4f}" + r"\ kJ/mol$" + "\n"
        r"$\bullet\ \Delta F_{Jar} = " + f"{delta_F:.4f}" + r"\ kJ/mol$" + "\n"
        r"$\bullet\ Satisfied:\ \langle W \rangle \geq \Delta F$"
    )
    ax3.text(0.1, 0.5, eq_text, transform=ax3.transAxes, fontsize=13, verticalalignment='center')

    # (4) 요약 테이블
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    table_data = [
        ["Sample Count", f"{n_samples}"],
        ["Landauer Limit (Theory)", f"{landauer_limit:.4f} kJ/mol"],
        ["Mean Work (<W>)", f"{avg_work:.4f} kJ/mol"],
        ["Jarzynski Free Energy", f"{delta_F:.4f} kJ/mol"],
        ["Dissipated Work (<W>-dF)", f"{dissipated_work:.4f} kJ/mol"]
    ]
    table = ax4.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.5, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # (5) 데이터 소스 목록
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    MAX_DISPLAY = 24
    if len(run_ids) > MAX_DISPLAY:
        display_ids = run_ids[:MAX_DISPLAY]
        footer_note = f"... and {len(run_ids) - MAX_DISPLAY} more runs."
    else:
        display_ids = run_ids
        footer_note = ""

    formatted_list = "Included Simulation Runs:\n"
    col_count = 0
    row_str = ""
    for rid in display_ids:
        row_str += f"{rid:<25} "
        col_count += 1
        if col_count >= 3:
            formatted_list += row_str + "\n"
            row_str = ""
            col_count = 0
    if row_str:
        formatted_list += row_str + "\n"
    
    formatted_list += footer_note

    ax5.text(0.5, 0.5, formatted_list, transform=ax5.transAxes, 
             fontsize=9, family='monospace', verticalalignment='center', horizontalalignment='center',
             bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=1'))

    plt.tight_layout()
    
    # 6. 저장 경로 설정 (날짜 + 시간 + 샘플수)
    # 현재 시간 가져오기 (년월일_시분)
    now = datetime.now()
    time_str = now.strftime("%y%m%d_%H%M") # 예: 251204_1720
    
    # 파일명 생성 (예: 251204_1720_14.png)
    save_filename = f"{time_str}_{n_samples}.png"
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    
    plt.savefig(save_path, dpi=150)
    print(f"[완료] 결과 그래프 저장됨: {save_path}")

if __name__ == "__main__":
    main()