# 本机构建说明（Windows + VS 2022 + CUDA）

> 针对本机环境的构建命令与踩坑记录。通用构建文档见 [docs/build.md](docs/build.md)。

## 环境概览

| 组件 | 路径 / 版本 |
|---|---|
| 构建目录 | `C:\llama.cpp\build`（已配置，勿随意删除） |
| 生成器 | Ninja（VS 自带）：`D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe` |
| CMake | VS 自带：`D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`（3.31.6） |
| 编译器 | MSVC 14.44（VS 2022 Community，D 盘） |
| CUDA | v13.3：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` |
| 当前配置 | Release、`GGML_CUDA=ON`、**`GGML_CUDA_FA_ALL_QUANTS=ON`**、`LLAMA_UI_GZIP=ON`（UI 从 HF bucket `ggml-org/llama-ui` 拉取） |

## 构建命令

**必须在 VS 开发者环境中编译**（否则 MSVC 找不到 `stdbool.h`/`windows.h`，报 C1083）。
vcvars 还需要 `vswhere.exe` 在 PATH 中（位于 `C:\Program Files (x86)\Microsoft Visual Studio\Installer`）。

最省事的方式：把下面内容存成 `build\_build.bat` 后执行 `cmd //c C:\llama.cpp\build\_build.bat`：

```bat
@echo off
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
call "D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build C:\llama.cpp\build --target llama-server -- -j 20
```

只改动了个别文件时，增量编译很快（几十秒）；改 `.cu` 文件才会触发 CUDA 重编（很慢）。

## 注意事项（踩坑记录）

1. **`-j auto` 不可用**：VS 自带的 ninja 1.12.1 不接受 `-j auto`/`-jauto`（报 `invalid -j parameter`），必须写显式数字。本机 20 个逻辑核，用 `-j 20` 全速编译。
2. **ninja 路径只有一个 `CMake` 目录**：`...\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe`，不是 `...\CMake\CMake\Ninja\`（后者是 `bin\cmake.exe` 所在的目录）。写错会报"系统找不到指定的路径"。
3. **`GGML_CUDA_FA_ALL_QUANTS=ON` 已开启**：编译所有 KV cache 量化组合的 FlashAttention kernel，支持任意 `-ctk/-ctv` 量化类型；代价是 CUDA 部分编译时间明显变长。如需修改：`cmake -B build -DGGML_CUDA_FA_ALL_QUANTS=OFF`（会触发 CUDA 全量重编）。
4. **UI assets 步骤每次构建都会跑**：`Provisioning UI assets` 的 stamp 文件从不生成，所以每次都会执行该步骤（npm 产物未变化时很快，直接跳过）。这是正常现象，不是构建卡住。
5. **ninja "只跑一步就退出 0" 不是故障**：UI assets 步骤带 `restat=1`，执行后若 `ui.cpp`/`ui.h` 未变化，其 dirty 状态被清除，下游编译/链接边随之变干净，ninja 正常退出。
6. **产物与部署**：
   - 产物在 `C:\llama.cpp\build\bin\`：`llama-server.exe`、`llama-server-impl.dll`、`mtmd.dll`、`llama.dll`、`ggml-*.dll` 等。
   - 部署到 `C:\llama\` 时需把相关 exe + dll 一起拷贝覆盖（尤其 `mtmd.dll`，视觉模型加载逻辑在里面）。
   - 只改 mtmd/clip 代码时，`mtmd.dll` 会更新但 `llama-server-impl.dll`/`llama-server.exe` 可能不重链接（导出符号未变、import 库未更新）——**运行时加载的是新的 `mtmd.dll`，功能正确**，但部署时仍建议把 `bin\` 下相关文件一起拷。
7. **不要杀正在运行的 llama-server 进程**（`C:\llama\llama-server.exe`）——当前会话就运行在它上面，杀掉会直接中断任务。构建输出在 `build\bin\`，与运行中的进程不冲突。
8. **Git Bash 下调用 cmd**：用 `cmd //c "..."`（双斜杠防止路径转换）；复杂引号场景建议写成 `.bat` 文件再调用，避免转义问题。

## 常用目标

```bat
cmake --build C:\llama.cpp\build --target llama-server -- -j 20   :: server（含 UI 嵌入）
cmake --build C:\llama.cpp\build --target llama-cli    -- -j 20   :: cli
cmake --build C:\llama.cpp\build --target llama-bench  -- -j 20   :: bench
cmake --build C:\llama.cpp\build --target mtmd         -- -j 20   :: 仅视觉/多模态库（改 clip.cpp 时最快）
cmake --build C:\llama.cpp\build --target all          -- -j 20   :: 全部
```
