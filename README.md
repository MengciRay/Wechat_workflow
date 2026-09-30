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

## 精读内容与篇幅

默认正文目标为 **3000–5000 字**，根据原文证据量调整；公众号简介仍限制为 110 字，不影响正文。
精读覆盖研究背景、方法细节、定量结果、关键图表、机制解释及局限六个方面，重点保留样本数、对照、工具、参数、单位、统计量及图表编号。
先逐段提取详细证据笔记，再综合成文，并对照全文 Markdown 核校术语、数字口径、图表与页码，删除跨片段造成的伪缺失和重复背景。若核校后正文少于 3000 个非空白字符、6 节或 18 段，会依据原笔记补充一次；仍不足则在预览中提示。篇幅检查只是粗略的完整性指标，不代表事实已核验，也不会为了凑字数补造原文没有的数据。
分段笔记输出预算为 8000 tokens，正文/全文校核各为 14000 tokens，因此精读比原先短摘要消耗更多时间和 API 额度。全文校核会将原始 Markdown 与初稿再次发送到 DeepSeek，超长文献可能触及所配置模型的上下文上限。
可以在结果 JSON 的 `notes` 中核查笔记，在同名 `.source.md` 中核查原始文字；`.md` 和 HTML 展示完整正文。

升级依赖并重新处理已有论文：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe workflow.py run
```

更新后的缓存配置会触发旧稿重新生成。若希望保留旧稿作比较，可先指定 `--output .\output\detailed`。

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
