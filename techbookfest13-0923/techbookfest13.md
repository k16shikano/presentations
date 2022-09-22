# 技術を本にして売る、という仕事

- 小川 晃通（著者）
- 鹿野 桂一郎（編集者）

# 自己紹介（小川晃通）

- 著書、共著、監訳書
    1. 『マスタリングTCP/IP RTP編』（オーム社）
    2. 『インターネットのカタチ』（オーム社）
    3. 『マスタリングTCP/IP OpenFlow編』（オーム社）
    4. 『アカマイ 知られざるインターネットの巨人』（KADOKAWA）
    5. 『ポートとソケットがわかればインターネットがわかる』（技術評論社）
    6. 『Linuxネットワークプログラミング』（ソフトバンククリエイティブ）
    7. 『プロフェッショナルIPv6 第2版』（ラムダノート）
    8. 『徹底解説v6プラス』（ラムダノート）
    9. 『ピアリング戦記 日本のインターネットを繋ぐ技術者たち』（ラムダノート）

# 自己紹介（鹿野桂一郎）

- オーム社で14年、書籍編集
    - 『マスタリングTCP/IP』シリーズ
    - 『型システム入門』
    - 『プログラミングのための線形代数』
- ラムダノートという出版社を立ち上げ
    - 新刊13冊 + 改訂2冊 + 1冊（β版）
    - 不定期刊行誌「n月刊ラムダノート」通巻6号

# ディスコグラフィー

- 企画の発端と制作システムを中心に

# 『マスタリングTCP/IP RTP編』

- 基本情報
    - Colin Perkins 著、小川 晃通 監訳
    - 2004年4月、オーム社
    - https://www.ohmsha.co.jp/book/9784274065613/

- 企画の背景と発端
    - 鹿野（出版社の編集者）が、翻訳の版権を獲得
    - 「RTPに詳しそうな人」を探して、あきみちさんに連絡

- 制作技術
    - 喫茶店で打ち合せしたり、休日に来社してもらったり、ふつうの商業出版
    - 業者による翻訳 → 版元で翻訳チェックと仮組版 → 監訳

# 『インターネットのカタチ』

- 基本情報
    - あきみち 著、空閑 洋平 著
    - 2011年6月、オーム社
    - https://www.ohmsha.co.jp/book/9784274068249/

- 企画の背景と発端
    - あきみちさんがブロガー @geekpage として独立
    - 「インターネットが壊れた話のネタがいろいろある」といって持ち込み

- 制作技術
    - TeXで執筆、Subversionのリモートサーバでバージョン管理

# 『マスタリングTCP/IP OpenFlow編』

- 基本情報
    - あきみち 著、宮永 直樹 著、岩田 淳 著
    - 2013年7月、オーム社
    - https://www.ohmsha.co.jp/book/9784274069208/

- 企画の背景と発端
    - 鹿野がSDNに興味を持って、あきみちさんと雑談してたのがきっかけ

- 制作技術
    - HTMLで執筆、Subversionのリモートサーバでバージョン管理
    - HTMLからPDFを自動組版するシステム（Scheme+LaTeX）

# 『プロフェッショナルIPv6』

- 基本情報
    - 小川 晃通 著
    - 2018年7月、ラムダノート
    - https://lambdanote.com/products/ipv6

- 企画の背景と発端
    - 鹿野が出版社 @lambdanote として独立
    - 「完成してない原稿をなんとかしたい」が、印税収入だと厳しい
        - 「もう本は書かないで」
    - クラウドファンディングでの出版に挑戦してみよう

- 制作技術
    - HTMLで執筆、GitHubでバージョン管理
    - HTMLからPDFを自動組版するシステム（Scheme+LaTeX）

# 『徹底解説v6プラス』

- 基本情報
    - 日本ネットワークイネイブラー株式会社 監修、小川 晃通・久保田 聡 共著
    - 2020年1月、ラムダノート
    - https://lambdanote.com/products/v6plus

- 企画の背景と発端
    - 『プロフェッショナルIPv6』のスポンサーでもあるJPNEさんから、あきみちさんに打診
    - サービスマニュアルやマーケ資料としてでなく、あくまでも「技術書」として企画
        - 大規模NAT技術について詳しい稀有な本に

- 制作技術
    - Markdownで執筆、GitHubでバージョン管理
    - MarkdownからPDFを自動組版するシステム（Haskell+LaTeX）

# 『プロフェッショナルIPv6 第2版』

- 基本情報
    - 小川 晃通 著
    - 2021年12月、ラムダノート
    - https://lambdanote.com/products/ipv6-2

- 追加のクラウドファンディングも企業スポンサーもなしで改訂に成功

# 『ピアリング戦記』

- 基本情報
    - 小川 晃通 著
    - 2022年7月、ラムダノート
    - https://lambdanote.com/products/peering

- 企画の背景と発端
    - ピアリング技術のコミュニティの中の方々（発起人）から、あきみちさんに打診
    - どういう本ができるかわからないけど、とりあえず当時を知っている人たちに順番に話を聞こう
    - 「本」としてまとまるまでは、あきみちさんも鹿野もそれぞれ苦労した

- 制作技術
    - Markdownで執筆、GitHubでバージョン管理
    - MarkdownからPDFを自動組版するシステム（Haskell+LaTeX）


# コンテンツのこと以外に、どんなことを考えているか

- 基本は「書きたいことがある → 本にする」だけど、本で生計を立てているとそうもいかない
    - クラウドファンディング
    - 企業/個人スポンサー
    - フリーミアム

- 出版には責任も伴う
    - 間違いがない内容にすることは前提
    - 無理に買わせない
    - パッケージ化された情報として残すことの意義

- バージョン管理と自動組版は空気と水のようなもの


## QA（後半）