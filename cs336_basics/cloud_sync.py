"""Google Drive 异步上传（可选）。

说明:
- 逻辑来自 Kaggle 上已验证的版本，行为保持一致；
- 没有 google 依赖、没有 token 文件、或未显式开启时，静默跳过，不影响本地训练；
- 上传在后台线程执行，不阻塞训练主循环。
"""

import os
import threading


def _resolve_run_folder(service, folder_id: str, run_name: str) -> str:
    """在目标目录下找到（或创建）以 run_name 命名的子文件夹。"""
    query = (
        f"name='{run_name}' and '{folder_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    found = service.files().list(q=query).execute().get("files", [])
    if found:
        return found[0]["id"]

    metadata = {
        "name": run_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [folder_id],
    }
    return service.files().create(body=metadata, fields="id").execute()["id"]


def _upload_task(local_path, cloud_filename, token_path, folder_id):
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        print(f"\n[云盘同步] 跳过：缺少依赖 ({exc})，可 pip install google-api-python-client\n")
        return

    try:
        creds = Credentials.from_authorized_user_file(
            token_path, ["https://www.googleapis.com/auth/drive"]
        )
        service = build("drive", "v3", credentials=creds)

        run_name = cloud_filename.split("/")[0]
        actual_filename = cloud_filename.split("/")[-1]

        target_folder_id = _resolve_run_folder(service, folder_id, run_name)

        media = MediaFileUpload(local_path, resumable=True)
        query = (
            f"name='{actual_filename}' and '{target_folder_id}' in parents and trashed=false"
        )
        existing = service.files().list(q=query).execute().get("files", [])

        print(f"\n[云盘同步] 正在上传 {actual_filename}...")
        if existing:
            service.files().update(fileId=existing[0]["id"], media_body=media).execute()
        else:
            metadata = {"name": actual_filename, "parents": [target_folder_id]}
            service.files().create(body=metadata, media_body=media).execute()
        print(f"[云盘同步] ✅ {actual_filename} 上传成功！\n")
    except Exception as exc:  # 网络/配额/权限问题都不应中断训练
        print(f"\n[云盘同步] ❌ 上传失败 [{cloud_filename}]: {exc}\n")


def upload_async(local_path, cloud_filename, token_path, folder_id) -> bool:
    """后台线程上传 local_path 到 Drive。返回是否真的启动了上传。"""
    if not (local_path and cloud_filename and folder_id):
        return False
    if not token_path or not os.path.exists(token_path):
        print(f"[云盘同步] 跳过：找不到凭据 {token_path}")
        return False

    threading.Thread(
        target=_upload_task,
        args=(local_path, cloud_filename, token_path, folder_id),
        daemon=True,
    ).start()
    return True
