import webview
import subprocess
import time
import os
import sys


# 1. 定義啟動 Streamlit 的函式
def run_streamlit():
    # 確保抓取正確的檔案路徑
    script_path = os.path.join(os.path.dirname(__file__), "stock_logic.py")
    # 啟動命令：headless 模式不開啟瀏覽器
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", script_path,
        "--server.headless", "true",
        "--server.port", "8501"
    ])


if __name__ == "__main__":
    print("正在啟動 AI 股市分析系統...")
    # 啟動後台服務
    process = run_streamlit()

    # 等待服務就緒
    time.sleep(5)

    try:
        # 建立獨立視窗
        webview.create_window(
            'AI 股市分析預測 App',
            'http://localhost:8501',
            width=1280,
            height=900,
            confirm_close=True
        )
        webview.start()
    except Exception as e:
        print(f"啟動失敗: {e}")
    finally:
        # 關閉視窗時自動殺掉後台進程
        process.terminate()
        sys.exit()
