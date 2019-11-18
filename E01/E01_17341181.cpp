#include<bits/stdc++.h>
using namespace std;
char data[100][100];
int start_x, start_y, end_x, end_y;
int min_distance = 999999;
bool visit[100][100];
vector< pair<int, int> > step;
vector< pair<int, int> > min_path;
void dfs(int x, int y, int length, vector< pair<int, int> > &path)
{
	visit[x][y] = 1;
	path.push_back(make_pair(x, y));
//	find the end, update min_length and min_path
	if(x == end_x && y == end_y)
	{
		if(length < min_distance)
		{
			min_distance = length;
			min_path = path;
		}
		return;
	}
//	search up, left, right, down
	for(int i=0; i<step.size(); i++)
	{
		int next_x = x + step[i].first;
		int next_y = y + step[i].second;
  		if(data[next_x][next_y] != '1' && visit[next_x][next_y] != 1)
  		{
   			length++;
   			dfs(next_x, next_y, length, path);
   			length--;
   			visit[next_x][next_y] = 0;
   			path.pop_back();
  		}  
 	}
	return;
}
int main()
{
	int m = 1, n;
	ifstream input("1.txt");
	step.push_back(make_pair(-1, 0));
	step.push_back(make_pair(0, -1));
	step.push_back(make_pair(0, 1));
	step.push_back(make_pair(1, 0));
	char s[100];
	input.getline(s, 100);
	n = strlen(s);
	strcpy(data[0], s);
	while(!input.eof())
	{
		visit[m][0] = 1;
		visit[m][n-1] = 1;
		for(int j=0; j<n; j++)
		{
			input>>data[m][j];
			if(data[m][j] == 'S') 
			{
				start_x=m;
				start_y=j;
			}
			if(data[m][j] == 'E')
			{
				end_x=m;
				end_y=j;
			}		
		}
		m++;
	}
	m--;
	input.close();
	vector< pair<int, int> > path;
	dfs(start_x, start_y, 0, path);
	if(min_distance == 999999) cout<<"This maze has not solution.\n";
	else
	{
		cout<<"The shortest path's length is: "<<min_distance<<endl;
		cout<<"The solution is:\n";
		for(int i=1; i<min_path.size()-1; i++) data[min_path[i].first][min_path[i].second] = '*';
		
		for(int i=0; i<m; i++)
		{
			for(int j=0; j<n; j++)
			{
				cout<<data[i][j];		
			}
			cout<<endl;
		}
	}
	return 0;
}
