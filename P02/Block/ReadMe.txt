blocksworld.pl是主程序，另外五个pl文件包含测试数据。
通过运行blocksworld.pl，输入case(n).即可进行测试，其中n = 1 to 5。
输出包括第一行为步数n，接下来n行为每一步的步骤，最后一行为能否达到目标状态，是则true，否则false。
其中样例5时间比较久，另外四个比较快。
运刚行的时候会报很多warning是因为注释的原因（但是我不知道为什么注释会报warning）。
开一个blocksworld.pl重复运行多个case会报warning的原因是重复定义了block和place。