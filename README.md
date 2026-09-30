# 文献 PDF → 公众号草稿箱

把可提取文字的论文 PDF 放入 `inbox`，程序使用 `pymupdf4llm.to_markdown(page_chunks=True)` 逐页提取 Markdown（保留标题、列表和可识别表格），调用 DeepSeek Flash 形成有页码依据的中文解读，并生成本地 HTML 预览。配置公众号接口后，可以把文章保存到**草稿箱**；程序不会执行群发或正式发布。

默认编辑侧重：单细胞组学、通用生物型人工智能（GBAI）和环境建模。生成内容需要人工核对，尤其是图表、公式、统计结果、引文和模型推断。扫描版 PDF 请先 OCR。

## 安装

需要 Python 3.10 以上。在本目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写 `DEEPSEEK_API_KEY`。默认模型名为 `deepseek-flash`。如果要上传公众号草稿，再填写 `WECHAT_APP_ID` 和 `WECHAT_APP_SECRET`，并在公众号后台确认草稿箱、素材接口权限及服务器出口 IP 白名单。`.env` 已在 `.gitignore` 中排除；不要把密钥发到聊天或提交到代码仓库。

## 处理 PDF

把 PDF 放进 `inbox` 后运行：

```powershell
.\.venv\Scripts\python.exe workflow.py run
```

如需持续监听文件夹，启动下面的命令并保持终端运行。复制完成的新 PDF 会自动处理，按 `Ctrl+C` 停止：

```powershell
.\.venv\Scripts\python.exe workflow.py watch
```

结果写入 `output`：同名 `.html` 是带核对提示的本地预览，`.wechat.html` 只含排版后的正文，可在浏览器中打开后复制到公众号编辑器，`.json` 保存结构化文章、逐段证据笔记、生成配置和篇幅统计；`.md` 是可编辑的详细解读；`.source.md` 是带 PDF 页码的原始提取文本。每个正文段落附有论文 PDF 的页码。再次运行时，相同 PDF、模型、编辑侧重、解析器版本及提示词配置会复用结果（旧版摘要缓存自动失效），避免重复消耗 API 额度。更改 `.env` 中的 `ARTICLE_FOCUS` 可以调整关注点；`--force` 强制重新生成。

也可以只处理一篇：

```powershell
.\.venv\Scripts\python.exe workflow.py prepare .\inbox\paper.pdf
```

## 解读内容与篇幅

正文默认目标 **1200–1800 字**，以主要发现与结论为中心；开头新增 **80–120 字的核心总结**，直接说明研究了什么、发现了什么、得出什么结论。
方法部分最多一小段，保留对象、数据范围和总体研究思路，省略工具版本、参数清单和操作步骤。主要发现保留代表性数字、必要的证据和局限；图表证据融入相关结果，不再逐图逐表展开。
术语以中文叙述为主；与主要发现相关的模型名称、专有算法和关键技术方法保留原文英文名称或通用缩写，首次出现时可简要说明用途。普通名词不逐个附英文，也不反复中英对照。

程序检查正文不超过 2000 个非空白字符、开头总结不超过 140 字符、方法部分最多 1 段且不超过 180 字符（英文、数字和标点也计入）。
超长、缺少总结或总结页码时会调整一次；仍不符合要求则报错，不会静默导出超长稿或直接截断句子。证据少时可更短，不再因不足章节数、段数或最低字数而扩写。

证据笔记仍保留详细信息，方便核查，但不要求全部搬入正文。先生成短稿，再对照全文 Markdown 核校，核校时同样遵守简洁要求。
`summary` 是带页码的开头总结，展示在 HTML、公众号正文和 Markdown 的标题之后；`digest` 仍是公众号卡片简介，不会重复插入正文。旧版不含 `summary` 的稿件仍可正常打开和上传。
分段笔记输出预算为 8000 tokens，正文/全文校核各为 6500 tokens；超长原文仍可能触及模型上下文上限。

新写作规则会使旧生成缓存失效。重新生成已有 PDF：

```powershell
.\.venv\Scripts\python.exe workflow.py prepare .\inbox\paper.pdf
```

如需与旧稿比较，可用 `--output .\output\concise` 单独保存新版本。已上传的微信草稿不会自动更新，需对新 JSON 另行运行 `draft`。

## 保存到公众号草稿箱

确认 `.env` 中公众号配置后，可对已经生成的文章执行：

```powershell
.\.venv\Scripts\python.exe workflow.py draft .\output\文章文件名.json
```

或者让 `inbox` 中的论文直接生成并保存草稿：

```powershell
.\.venv\Scripts\python.exe workflow.py run --draft
```

持续监听并自动存草稿：

```powershell
.\.venv\Scripts\python.exe workflow.py watch --draft
```

程序默认生成一张纯图案封面并上传为永久素材。如需自己的封面：

```powershell
.\.venv\Scripts\python.exe workflow.py run --draft --cover .\cover.png
```

也可以在 `.env` 设置已有的永久图片素材 ID：`WECHAT_COVER_MEDIA_ID`。草稿成功后会生成 `.draft.json` 回执，重复运行会跳过同一篇文章的重复上传。需要新草稿时使用 `--force`。如果公众号接口没有权限，命令会报告微信返回的错误码；本地 HTML 仍可用于复制排版。

## 微信 IP 白名单与代理

微信请求（获取 token、封面上传和草稿接口）默认直连，避免系统代理切换出口 IP 导致 `40164 not in whitelist`。
将微信错误信息中实际报告的出口公网 IP 加入该公众号的 API IP 白名单，保留已有条目。
若网络必须通过系统代理访问微信，可在 `.env` 中设置 `WECHAT_USE_SYSTEM_PROXY=1`，并将代理的固定出口 IP 加入白名单。
默认直连仅避开代理出口；若直连网络本身使用动态公网 IP 或多个 NAT 出口，仍需配置相应白名单或使用固定出口。

## 处理范围与限制

- 全文文字会分段送到 DeepSeek API；请确认你有权将相应文献内容交给该服务处理。
- PDF 页码指 PDF 文件中的页序，与印刷版页码可能不同。
- `pymupdf4llm` 将文字层和可识别的版面结构转为 Markdown；复杂双栏、表格、公式仍可能有提取误差。当前显式关闭自动 OCR，扫描件请先 OCR。图像不会发送给模型，图表解读只依据可提取的正文、表格及图注，不能等同于视觉读图。几乎无文字时程序会停止；混合文档中的空白提取页保持原页码并在稿件中提醒核对。
- 草稿箱和永久素材接口的可用性取决于公众号账号权限与微信平台配置。草稿保存成功也不代表已经发布或推送给粉丝。
- AI 生成的解读只适合做初稿，正式发布前请对照论文核对结论和引用。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 接口依据

- [PyMuPDF4LLM Markdown / page_chunks API](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html)

- [DeepSeek 模型与标识](https://api-docs.deepseek.com/quick_start/pricing/)
- [DeepSeek JSON 输出](https://api-docs.deepseek.com/guides/json_mode/)
- [微信草稿箱：新增草稿](https://developers.weixin.qq.com/doc/offiaccount/Draft_Box/Add_draft.html)
- [微信永久素材：新增永久素材](https://developers.weixin.qq.com/doc/offiaccount/Asset_Management/Adding_Permanent_Assets.html)
