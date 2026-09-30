import time
import subprocess
import argparse
import datetime as dt

def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def cancel_job(job_id, delay_minutes):
    """
    在指定的分钟数后取消一个 SLURM 作业。
    
    :param job_id: 要取消的 SLURM 作业 ID
    :param delay_minutes: 延迟的时间（分钟）
    """
    print(f"[{now_str()}]", flush=True)

    delay_seconds = delay_minutes * 60
    print(f"将在 {delay_minutes} 分钟后取消作业 {job_id}...")
    time.sleep(delay_seconds)
    
    try:
        subprocess.run(["scancel", str(job_id)], check=True)
        print(f"✅ 成功取消作业 {job_id}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 取消作业 {job_id} 失败: {e}")
    
    print(f"[{now_str()}]", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--delay_minutes", required=True, type=int)
    args = parser.parse_args()
    cancel_job(args.job_id, args.delay_minutes)