import requests

url = "https://api.ai21.com/studio/v1/segmentation"

payload = {
    "source": '''题⽬描述
赛题： AIGC 相关的产品开发 + 关键词 Accessible
题⽬释义
今年，以 ChatGPT、Midjourney 等为代表的⽣成式 AI 产品席卷全球互联⽹，在探索全新的产品形态与应⽤场景 的同时，我们也在关注着 AI 能否帮助⼈类深⼊到 曾经不可及的⻆落，让科技可以触达每 ⼀个个体，创造出独特的 ⽤户体验。
本题为开放性题⽬，以下是⼏个可以思考的⽅向示例：
能够为特定⼈群提供特殊帮助
能够触及更多⼈群与场景
能够引发社会思考，传递社会价值
能够带来全新的体验
作品评分标准参考
创新性 40%
完成度 20%
技术⽔平 20%
⽤户体验 20%
我们会根据以上四个维度综合参考确认进⼊线下路演答辩的组别名单，同时结合线下路演答辩的表现给出最终评
分，并评选出各奖项的获得者
项⽬举例
1. AIGC 赋能的与主题相关的游戏开发
2. 利⽤、集成 AIGC 技术落地⼀个与主题相关的产品
3. 思考 AIGC 的未来，探讨设计⼀个有深度的产品概念
作品提交
选⼿须提交：
可运⾏、供概念验证的技术 demo（并附上相应项⽬源码）
简短的对作品技术概念的基本说明、功能介绍与演示录屏等⽂档与资料，帮助我们了解你的作品
提交⽅式：
选⼿为项⽬创建⼀个 GitHub 仓库，在仓库中进⾏开发并保留所有的 commit 记录，主办⽅将选择仓库中 7 ⽉ 5 ⽇ 18 点前提交的最后⼀个 commit 作为最终上交作品。''',
    "sourceType": "TEXT"
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": "Bearer bK1MWxFztgAmAkQxeG0Ccn4VZJ50ZI6P"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)