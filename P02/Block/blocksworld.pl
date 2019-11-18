%定义位置和木块
object(X):- place(X); block(X).

%可以move的条件
can(move(Block,From,To), [clear(Block), on(Block,From), clear(To)]) :- block(Block), object(To), object(From), To\=Block, From\=To, From\=Block.

%move的效果
adds(move(Block,From,To), [on(Block,To),clear(From)]).
deletes(move(Block,From,To), [on(Block,From), clear(To)]).

%到达该状态时，添加效果
achieves(Action,Goal) :- adds(Action,Goals), member(Goal,Goals).
%到达该状态时，删除效果
preserves(Action,Goals) :- deletes(Action,Relations), not((member(Goal,Relations), member(Goal,Goals))).
%回溯，先求出新加入的关系，然后在目标状态中删除新加入的关系中含有的关系，求出差集，再求出当前action的条件，加入到差集中，生成RegressGoals，即新的目标
regress(Goals,Action,RegressGoals) :- adds(Action,NewRelations), delete_all(Goals,NewRelations,RestGoals), can(Action,Condition), add_new_relation(Condition,RestGoals,RegressGoals).

%是否无解（即condition是不可满足的）
impossible(on(X,X),_).
impossible(on(X,Y),Goals) :- member(clear(Y),Goals); member(on(X,Y1),Goals),Y1\=Y; member(on(X1,Y),Goals),X1\=X.
impossible(clear(X),Goals) :- member(on(_,X),Goals).

%在状态中增加某些关系
add_new_relation([],L,L).
add_new_relation([Goal|_],Goals,_) :- impossible(Goal,Goals),!,fail.
add_new_relation([X|L1],L2,L3) :- member(X,L2), !, add_new_relation(L1,L2,L3).
add_new_relation([X|L1],L2,[X|L3]) :- add_new_relation(L1,L2,L3).

%删除状态中在L2中的关系，并求Diff差集
delete_all([],_,[]).
delete_all([X|L1],L2,Diff) :- member(X,L2), !, delete_all(L1,L2,Diff).
delete_all([X|L1],L2,[X|Diff]) :- delete_all(L1,L2,Diff).

%自定义操作符#，表示连接状态和动作
:- op(400,yfx,#).

%找后继，通过action来找，每个后继对应一个新的目标，为根据f值去搜索每个后继做准备
succ(Goals # _, NewGoals # Action,1) :- member(Goal,Goals), can(Action,_), achieves(Action,Goal), preserves(Action,Goals), regress(Goals,Action,NewGoals).

%通过删除来判断是否达到目标状态
goal(Goals # _) :- start(State), satisfied(State,Goals).
satisfied(State,Goals) :- delete_all(Goals,State,[]).

%求启发值
h(Goals # _,H) :- start(State), calculate_h(Goals,State,H).
calculate_h(A,B,K) :- ((setof(X,(wrong_pos(X,A,B)),R))->length(R,K1);(K1 = 0,R = []) ),(setof(D,(member(D,R),under(A,E,D),under(B,E,D)),R2)->length(R2,K2);K2 = 0),K is K1 + K2.
%用于求启发值的谓词，表示不在目标位置的木块个数
wrong_pos(X,A,B) :- member(on(X,Y),A), place(Y), member(on(X,Z),B), Z\=Y.
wrong_pos(X,A,B) :- member(on(X,Y),A), block(Y), member(on(X,Z),B), (Z\=Y;wrong_pos(Y,A,B)).
%用于求启发值的谓词，表示不在目标位置但下方有在目标位置的木块个数
under(A,Y,X):- member(on(X,Y),A), block(Y).
under(A,Y,X):- member(on(X,Z),A), block(Z), under(A,Y,Z).

%search为主函数，End为目标状态，Solution为解
search(End,Solution) :- expand([],l(End,0/0),99999,_,yes,Solution).

%expand为扩展结点的函数，递归
%如果表头是目标，则倒数第二个参数为yes，表示查到解了
expand(P,l(N,_),_,_,yes,[N|P]) :- goal(N).
%第一步扩展（优先队列为空的时候的扩展）
expand(P,l(N,F/G),Bound,Tree1,Solved,Sol) :- F=<Bound, (bagof(M/C,(succ(N,M,C), \+member(M,P)),Succ), !, succlist(G,Succ,Ts), min_f(Ts,F1), expand(P,t(N,F1/G,Ts), Bound, Tree1, Solved,Sol); Solved == never).
%取后继里面最小f值的进行扩展（continues），当扩展完这个结点就回溯去扩展下一个结点
expand(P,t(N,F/G,[T|Ts]),Bound,Tree1,Solved,Sol) :- F =< Bound, min_f(Ts,BF),min(Bound,BF,Bound1), expand([N|P],T,Bound1,T1,Solved1,Sol), continues(P,t(N,F/G,[T1|Ts]), Bound, Tree1, Solved1, Solved, Sol).
%无后继则截断递归并令Solved为never以控制回溯
expand(_,t(_,_,[]),_,_,never,_):-!.
expand(_,Tree,Bound,Tree,no,_):- f(Tree,F),F>Bound.
%对某结点进行扩展的函数，将后继中最小f值的结点插入优先队列
continues(_,_,_,_,yes,yes,Sol).
continues(P,t(N,F/G,[T1|Ts]),Bound,Tree1,no,Solved,Sol):- insert(T1,Ts,NTs), min_f(NTs,F1), expand(P,t(N,F1/G,NTs),Bound,Tree1,Solved,Sol).
continues(P,t(N,F/G,[_|Ts]),Bound,Tree1,never,Solved,Sol):- min_f(Ts,F1), expand(P,t(N,F1/G,Ts),Bound,Tree1,Solved,Sol).

%数据结构优先队列，按照f=g+h来排序
succlist(_,[],[]).
succlist(G0,[N/C|NCs],Ts):- G is G0 + C, h(N,H), F is G+H, succlist(G0,NCs,Ts1), insert(l(N,F/G),Ts1,Ts).
%优先队列的插入动作
insert(T,Ts,[T|Ts]) :- f(T,F),min_f(Ts,F1), F=<F1,!.
insert(T,[T1|Ts],[T1|Ts1]) :- insert(T,Ts,Ts1).

f(l(_,F/_),F).
f(t(_,F/_,_),F).
min_f([T|_],F) :- f(T,F).
min_f([],9999).

%求X和Y的最小值M
min(X,Y,M):- X < Y -> M is X; M is Y.

%将Solution中的每一步动作输出
print_action([]).
print_action(Plan):- [A|B] = Plan, X#Y = A, write(Y), nl, print_action(B).

%测试样例
case(N):- N == 1, consult('./case1.pl'), search([on(b3,b1),on(b1,1),on(b2,2),clear(b3),clear(b2)]#end,Plan), length(Plan,L1), L is L1-1, write(L), nl, print_action(Plan).
case(N):- N == 2, consult('./case2.pl'), search([on(b2,b1),on(b1,b3),on(b3,2),on(b4,b5),on(b5,4)]#end,Plan), length(Plan,L1), L is L1-1, write(L), nl, print_action(Plan).
case(N):- N == 3, consult('./case3.pl'), search([on(b4,b3),on(b3,b5),on(b5,b1),on(b1,b2),on(b2,2)]#end,Plan), length(Plan,L1), L is L1-1, write(L), nl, print_action(Plan).
case(N):- N == 4, consult('./case4.pl'), search([on(b6,b2),on(b2,b4),on(b4,b1),on(b1,b3),on(b3,b5),on(b5,1)]#end,Plan), length(Plan,L1), L is L1-1, write(L), nl, print_action(Plan).
case(N):- N == 5, consult('./case5.pl'), search([on(b7,b2),on(b2,b4),on(b4,b1),on(b1,b3),on(b3,b6),on(b6,b8),on(b8,b5),on(b5,1)]#end,Plan), length(Plan,L1), L is L1-1, write(L), nl, print_action(Plan).