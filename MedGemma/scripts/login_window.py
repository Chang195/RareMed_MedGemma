"""Local masked HF login; never reads clipboard or prints credentials."""
import os
from pathlib import Path
import os
import threading
import tkinter as tk
from tkinter import ttk
os.environ['HF_HOME'] = str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from huggingface_hub import login
from huggingface_hub.utils import logging
logging.set_verbosity_error()

window = tk.Tk()
window.title('RareMed — 本机 Hugging Face 登录')
window.geometry('650x265')
ttk.Label(window, text='在下方输入框按 Ctrl+V 粘贴只读 Token。星号表示已输入。').pack(pady=14)
value = tk.StringVar()
entry = ttk.Entry(window, textvariable=value, show='*', width=72)
entry.pack(padx=18)
status = tk.StringVar(value='已输入 0 个字符')
ttk.Label(window, textvariable=status, wraplength=610).pack(pady=12)
ttk.Label(window, text='凭据仅保存到 G:\\RareMed_MedGemma_Local\\huggingface；不加入 Git 凭据。').pack()

def changed(*_):
    status.set(f'已输入 {len(value.get())} 个字符（内容已隐藏）')
value.trace_add('write', changed)

def finish(error):
    button.config(state='normal')
    entry.config(state='normal')
    value.set('')
    if error is None:
        status.set('登录成功。Token 已保存至本机。可以关闭窗口并通知 Codex 继续验证模型访问。')
    else:
        status.set(f'登录失败：{error}。若为 HfHubHTTPError，请检查 Token 是否有效；网络错误请检查连接。')

def submit():
    token = value.get().strip()
    if not token.startswith('hf_') or any(c.isspace() for c in token):
        status.set('输入不是预期的 hf_ 开头 Token，或含空白。请重新复制 Token 本身。')
        return
    entry.config(state='disabled'); button.config(state='disabled')
    status.set('正在连接 Hugging Face 验证；凭据不会显示或写入日志。')
    def worker():
        error = None
        try:
            login(token=token, add_to_git_credential=False)
        except Exception as exc:
            error = type(exc).__name__
        window.after(0, lambda: finish(error))
    threading.Thread(target=worker, daemon=True).start()

button = ttk.Button(window, text='验证并登录', command=submit)
button.pack(pady=14)
entry.focus_set()
window.mainloop()
