#include<iostream>
#include<stdlib.h>
#include<vector>
#include<limits.h>
#include <bits/stdc++.h>
using namespace std;

struct task_st 
{
    int task_number;            //task number
    float finish_time_L;        //finish time on local core
    float finish_time_WS;       //finish time on wireless sending channel
    float finish_time_C;        //finish time on cloud
    float finish_time_WR;        //finish time of receiving back the result
    float start_time;          //start time of task
    float ready_time_L;
    float ready_time_WS;
    float ready_time_C;
    float ready_time_WR;
    bool is_cloud;          //1 if task is a cloud task
    bool is_initialTask;
    bool is_exitTask;
    int core_No;            //core number
    float priority;        //priority of task
    float com_cost;        //computation cost 
    std::vector<int> successors;    //successors of each task
    std::vector<int> predecessors;  //predecessors of each task
    std::vector<int> core_time;     //execution time on each core
};
std::vector<task_st> task;
std::vector<task_st> task_temp;

//vector for storing task and the task its pointed to
std::vector<std::vector<int>> tasks_edge;

//vector for storing the power consumption on each core
std::vector<float> power;

//vector to store the schedule on each core and cloud (at 0 index)
std::vector<float> schedule;
std::vector<float> schedule_temp;

//vectors to store sorted priority and task numbers
std::vector<float> sorted_priority;
std::vector<int> sorted_task_number;

//power_c = cloud consumption
//n_c = number of cores
//n_t = number of tasks
//n_e = number of edges in the task graph
//t_s = time for sending to cloud
//t_c = time for cloud computing
//t_r = time for sending back the result from cloud
//T_re = total time (t_s + t_c + t_r)
int n_c, n_t, n_e; 
float power_c,t_s, t_c, t_r, T_re, total_Time, T_max, initial_Energy;
float schedule_WS = 0, schedule_C = 0, schedule_WR = 0;
float schedule_WS_temp = 0, schedule_C_temp = 0, schedule_WR_temp = 0;

void print_taskVector(std::vector<task_st> *task1)
{
    cout << "*****************************************************************************************" << endl;
    int i, j, k;
    for(i = 1; i < (*task1).size(); i++)
    {
        k = sorted_task_number[i];
        cout << "Task : " << (*task1)[k].task_number << "  -->  ";
        
        if((*task1)[k].is_cloud)
        {
            cout << "Executed on Cloud" << "  -->  ";
            cout << "Ready Time WS  : " << (*task1)[k].ready_time_WS << "  -->  ";
            cout << "Finish Time WS   : " << (*task1)[k].finish_time_WS << endl;
            cout << "\t\t\t\t   Ready Time C   : " << (*task1)[k].ready_time_C << "  -->  ";
            cout << "Finish Time C    : " << (*task1)[k].finish_time_C << endl;
            cout << "\t\t\t\t   Ready Time WR  : " << (*task1)[k].ready_time_WR << "  -->  ";
            cout << "Finish Time WR   : " << (*task1)[k].finish_time_WR << endl;
        }
        else
        {
            cout << "Executed on Core " << (*task1)[k].core_No << "  -->  ";
            cout << "Ready Time     : " << (*task1)[k].ready_time_L << "  -->  ";
            cout << "Finish Time      : " << (*task1)[k].finish_time_L << endl;
        }
        cout << endl;
    }
    cout << "*****************************************************************************************" << endl;
}


float completion_Time(std::vector<task_st> *task1)
{
    float time_consumed = 0;
    int i, j;
    for ( i = 1; i < (*task1).size(); i++ )
    {
        time_consumed = max(time_consumed, max((*task1)[i].finish_time_L, (*task1)[i].finish_time_WR));
    }
    return time_consumed;
}

float energy_Consumption(std::vector<task_st> *task1)
{
    float energy_L = 0, enerygy_C = 0, total_Energy = 0;
    int i;
    for( i = 1; i < (*task1).size(); i++)
    {
        //if the task i is not a cloud task
        if((*task1)[i].is_cloud == 0)
        {
            energy_L = energy_L + (power[(*task1)[i].core_No] * (*task1)[i].core_time[(*task1)[i].core_No]);
        }
        else if ((*task1)[i].is_cloud == 1)
        {
            enerygy_C = enerygy_C + (power[(*task1)[i].core_No] * t_s);
        }
    }
    total_Energy = enerygy_C + energy_L;
    return total_Energy;
}

float calculate_readyTime_WS(int task_No, std::vector<task_st> *task1)
{
    int pre_task_No = (*task1)[task_No].predecessors[0];
    float max_readyTime = max((*task1)[pre_task_No].finish_time_L, (*task1)[pre_task_No].finish_time_WS);
    for (int k = 1; k < (*task1)[task_No].predecessors.size(); k++)
    {
        pre_task_No = (*task1)[task_No].predecessors[k];
    
        if (max_readyTime < max((*task1)[pre_task_No].finish_time_L, (*task1)[pre_task_No].finish_time_WS))
        {
            max_readyTime = max((*task1)[pre_task_No].finish_time_L, (*task1)[pre_task_No].finish_time_WS);
        }
    }
return max_readyTime;
}

float calculate_readyTime_C(int task_No, std::vector<task_st> *task1)
{
    int pre_task_No = (*task1)[task_No].predecessors[0];
    float max_readyTime_C = (*task1)[pre_task_No].finish_time_C;
    for (int k = 1; k < (*task1)[task_No].predecessors.size(); k++)
    {
        pre_task_No = (*task1)[task_No].predecessors[k];
        
        if (max_readyTime_C < (*task1)[pre_task_No].finish_time_C)
        {
            max_readyTime_C = (*task1)[pre_task_No].finish_time_C;
        }
    }
return max((*task1)[task_No].finish_time_WS, max_readyTime_C);
}

float calculate_readyTime_L(int task_No, std::vector<task_st> *task1)
{
    if((*task1)[task_No].predecessors.size() == 0)
    {
        return 0;
    }
    int pre_task_No = (*task1)[task_No].predecessors[0];
    float max_readyTime_L = max((*task1)[pre_task_No].finish_time_L, (*task1)[pre_task_No].finish_time_WR);
    for (int k = 1; k < (*task1)[task_No].predecessors.size(); k++)
    {
        pre_task_No = (*task1)[task_No].predecessors[k];
        if(max_readyTime_L < max((*task1)[pre_task_No].finish_time_L, (*task1)[pre_task_No].finish_time_WR))
        {
            max_readyTime_L = max((*task1)[pre_task_No].finish_time_L, (*task1)[pre_task_No].finish_time_WR);
        }
    }
return max_readyTime_L;
}


void calculate_finshTime_Cloud(int task_No, std::vector<task_st> *task1, float schedule_WS1, float schedule_C1, float schedule_WR1)
{
    //calculating ready time of task.
    //ready time for wireless sending channel
    if((*task1)[task_No].predecessors.size() == 0)
    {
        (*task1)[task_No].ready_time_WS = schedule_WS1;
        (*task1)[task_No].finish_time_WS = (*task1)[task_No].ready_time_WS + t_s;
        (*task1)[task_No].ready_time_C = (*task1)[task_No].finish_time_WS;
        (*task1)[task_No].finish_time_C = (*task1)[task_No].ready_time_C + t_c;
        (*task1)[task_No].ready_time_WR = (*task1)[task_No].finish_time_C;
    }
    else
    {
        (*task1)[task_No].ready_time_WS = max(calculate_readyTime_WS(task_No, task1), schedule_WS1);
        (*task1)[task_No].finish_time_WS = (*task1)[task_No].ready_time_WS + t_s;
        (*task1)[task_No].ready_time_C = max(calculate_readyTime_C(task_No, task1), schedule_C1);
        (*task1)[task_No].finish_time_C = (*task1)[task_No].ready_time_C + t_c;
        (*task1)[task_No].ready_time_WR = max((*task1)[task_No].finish_time_C, schedule_WR1);
    }

    //finish time for transmitting back the result.
    (*task1)[task_No].finish_time_WR = (*task1)[task_No].finish_time_C + t_r;
    (*task1)[task_No].core_No = 0;
    (*task1)[task_No].finish_time_L = 0;
    (*task1)[task_No].ready_time_L = 0;
}

//scheduling each task on cloud if it was originally   
//scheduled on any of the local core and simultaneously
//scheduling each task on other cores other than its original core.
//and considering
//the combination which has minimum energy among the three combinations
void task_migration()
{
    int i, j, k, l, p, n;
    float time_taken = 0, energy = 0;
    
    schedule_temp = {0,0,0,0};
    schedule_WR_temp = 0; //schedule_WR;
    schedule_C_temp = 0; //schedule_C;
    schedule_WS_temp = 0; //schedule_WS;
    float ratio_energy_time = (initial_Energy - energy) / (total_Time - time_taken);    
    
    for (i = 1; i < sorted_task_number.size(); i++)
    {
        //i = sorted_task_number[l];
        task_temp = task;
        schedule_temp = {0,0,0,0};
        schedule_WR_temp = 0; //schedule_WR;
        schedule_C_temp = 0; //schedule_C;
        schedule_WS_temp = 0; //schedule_WS;

        if(task_temp[i].is_cloud != 1)
        {
            //scheduling each task on cloud if it was originally   
            //scheduled on any of the local core
            task_temp[i].is_cloud = 1;
            
            for (p = 1; p < sorted_task_number.size(); p++)
            {
                n = sorted_task_number[p];
                if(task_temp[n].is_cloud == 1)
                {
                    calculate_finshTime_Cloud(n, &task_temp, schedule_WS_temp, schedule_C_temp, schedule_WR_temp);
                    schedule_C_temp = task_temp[n].finish_time_C;
                    schedule_WS_temp = task_temp[n].finish_time_WS;
                    schedule_WR_temp = task_temp[n].finish_time_WR;
                    task_temp[n].finish_time_L = 0;
                    task_temp[n].ready_time_L = 0;
                }
                else
                {
                    int r = task_temp[n].core_No;
                    task_temp[n].ready_time_L = calculate_readyTime_L(n, &task_temp);  
                    if(task_temp[n].ready_time_L >= schedule_temp[r])
                    {
                        task_temp[n].finish_time_L = task_temp[n].core_time[r] + task_temp[n].ready_time_L;
                        schedule_temp[r] = task_temp[n].finish_time_L;
                    }
                    else
                    {
                        task_temp[n].ready_time_L = schedule_temp[r];
                        task_temp[n].finish_time_L = task_temp[n].core_time[r] + task_temp[n].ready_time_L;
                    }
                }
            }
            
            time_taken = completion_Time(&task_temp);
            energy = energy_Consumption(&task_temp);
            if (time_taken < T_max)
            {
                energy = energy_Consumption(&task_temp);
                if (energy <= initial_Energy)
                {
                    initial_Energy = energy;
                    task = task_temp;
                }
            } 
        }
    }
    
    for (i = 1; i < sorted_task_number.size(); i++)
    {
        task_temp = task;
        schedule_C_temp = 0;
        schedule_WS_temp = 0;
        schedule_WR_temp = 0;
        schedule_temp = {0,0,0,0};
        
        if(task_temp[i].is_cloud != 1)
        {
            int core = task_temp[i].core_No; 
            for (k = 1; k < task_temp[i].core_time.size(); k++)
            {
                //scheduling each task on other cores other than its original core.
                schedule_temp = {0,0,0,0};
                schedule_WR_temp = 0; //schedule_WR;
                schedule_C_temp = 0; //schedule_C;
                schedule_WS_temp = 0; //schedule_WS;
                    
                if (core != k)
                {
                    task_temp[i].core_No = k;
                    task_temp[i].finish_time_WR = 0;
                    task_temp[i].finish_time_C = 0;
                    task_temp[i].finish_time_WS = 0;
                    task_temp[i].ready_time_WR = 0;
                    task_temp[i].ready_time_C = 0;
                    task_temp[i].ready_time_WS = 0;
                
                    for (p = 1; p < sorted_task_number.size(); p++)
                    {
                        n = sorted_task_number[p];
                        if(task_temp[n].is_cloud == 1)
                        {
                            calculate_finshTime_Cloud(n, &task_temp, schedule_WS_temp, schedule_C_temp, schedule_WR_temp);
                            schedule_C_temp = task_temp[n].finish_time_C;
                            schedule_WS_temp = task_temp[n].finish_time_WS;
                            schedule_WR_temp = task_temp[n].finish_time_WR;
                        }
                        else
                        {
                            int r = task_temp[n].core_No;
                            task_temp[n].ready_time_L = calculate_readyTime_L(n, &task_temp);  
                            if(task_temp[n].ready_time_L >= schedule_temp[r])
                            {
                                task_temp[n].finish_time_L = task_temp[n].core_time[r] + task_temp[n].ready_time_L;
                                schedule_temp[r] = task_temp[n].finish_time_L;
                            }
                            else
                            {
                                task_temp[n].ready_time_L = schedule_temp[r];
                                task_temp[n].finish_time_L = task_temp[n].core_time[r] + task_temp[n].ready_time_L;
                            }
                        }
                    }
                    
                    time_taken = completion_Time(&task_temp);
                    energy = energy_Consumption(&task_temp);
                    if (time_taken < T_max)
                    {
                        energy = energy_Consumption(&task_temp);
                        if (energy <= initial_Energy)
                        {
                            initial_Energy = energy;
                            task = task_temp;
                        }
                    }
                }
            }
        }
    }
    time_taken = completion_Time(&task);
    energy = energy_Consumption(&task);
    print_taskVector(&task);
    cout << "Total completion time : " << time_taken << endl;
    cout << "Total energy comsumed : " << energy << endl;
}


void initial_scheduling()
{
    //section 3 : primary execution.
    //assigning cloud task based on minimum execution time
    //and calculating computation cost of each task.
    int i , j, sum, minTime;
    for (i = 1; i < task.size(); i++)
    {
        minTime = task[i].core_time[1];
        sum = task[i].core_time[1];
        for (j = 2; j < task[i].core_time.size(); j++)
        {
            if (minTime > task[i].core_time[j])
            {
                minTime = task[i].core_time[j];
            }
            sum = sum + task[i].core_time[j];
        }
        
        if (T_re < minTime)
        {
            task[i].is_cloud = 1;
            task[i].com_cost = (float) T_re;
        }
        else
        {
            task[i].is_cloud = 0;
            task[i].com_cost = (float) sum / (float) n_c;
        }
    }
/*   
    cout << "Computation cost of each task: " << endl;
    for (i = 1; i < task.size(); i++)
    {
     cout << i << " : " << setprecision(5) << task[i].com_cost << " " << endl;  
    }
*/  
    //section 3 : task prioritizing.
    //calculating priority of each task.
    int task_number;
    float max_succ_priority;
    
    for (i = task.size()-1; i > 0; i--)
    {
        task[i].start_time = 0;
        task[i].finish_time_L = 0;
        task[i].finish_time_WS = 0;
        task[i].finish_time_C = 0;
        task[i].finish_time_WR = 0;
        task[i].ready_time_L = 0;
        task[i].ready_time_WS = 0;
        task[i].ready_time_C = 0;
        task[i].ready_time_WR = 0;
        if (task[i].successors.size() == 0)
        {
            task[i].priority = task[i].com_cost;
        }
        
        else
        {
            task_number = task[i].successors[0];
            max_succ_priority = task[task_number].priority;
           
            for (j = 1; j < task[i].successors.size(); j++)
            {
                task_number = task[i].successors[j];
                if (max_succ_priority < task[task_number].priority)
                {
                    max_succ_priority = task[task_number].priority;
                }
            }
            task[i].priority = max_succ_priority + task[i].com_cost;
        }
//        cout << "priority of " << i << " is " << setprecision(5) << task[i].priority << endl;
    }
    
    //Execution Unit Selection
    
    //sorting the tasks according to their priority 
    //and storing the result in sorted_task_number vector
    sorted_priority.resize(n_t+1);
    sorted_task_number.resize(n_t+1);
    float temp_priority;
    int temp_task_number;
    for (i = 1; i < task.size(); i++)
    {
        sorted_task_number[i] = task[i].task_number;
        sorted_priority[i] = (float) task[i].priority;
    }
    for(i = 1; i < task.size(); i++)
    {
        for (j = i+1; j < task.size(); j++)
        {
            if (sorted_priority[i] < sorted_priority[j])
            {
                temp_priority = sorted_priority[j];
                sorted_priority[j] = sorted_priority[i];
                sorted_priority[i] = temp_priority;
                temp_task_number = sorted_task_number[j];
                sorted_task_number[j] = sorted_task_number[i];
                sorted_task_number[i] = temp_task_number;
            }
        }
    }
    cout << endl;
    cout << "*****************************************************************************************" << endl;
    cout << "\t TASKS IN PRIORITY ORDER " << endl;
    cout << "*****************************************************************************************" << endl;
    cout << "   " << "TASK No.\t" << "CALCULATED PRIORITY" << endl;
    for (i = 1; i < sorted_priority.size(); i++)
    {
        cout << "   " << sorted_task_number[i] << "\t" << sorted_priority[i] << endl;
    }
    cout << endl;

    //ft_l contains the finish time of each task on the given cores
    std::vector<float> ft_L(n_c);
    for (i = 1; i < sorted_task_number.size(); i++)
    {
        int task_No = sorted_task_number[i];
        if (task[task_No].is_cloud == 1)
        {
            calculate_finshTime_Cloud(task_No, &task, schedule_WS, schedule_C, schedule_WR);
            schedule_C = task[task_No].finish_time_C;
            schedule_WS = task[task_No].finish_time_WS;
            schedule_WR = task[task_No].finish_time_WR;
        }
        //if the task with task_No is not a cloud task
        else
        {
            task[task_No].ready_time_L = calculate_readyTime_L(task_No, &task);
            
            for (int k = 1; k < task[task_No].core_time.size(); k++ )
            {
                if (task[task_No].ready_time_L >= schedule[k])
                    ft_L[k-1] = task[task_No].ready_time_L + task[task_No].core_time[k];
                else
                    ft_L[k-1] = schedule[k] + task[task_No].core_time[k];
            }
            
            calculate_finshTime_Cloud(task_No, &task, schedule_WS, schedule_C, schedule_WR);
            float min_vectorValue = *min_element(ft_L.begin(), ft_L.end());
            float minExec_Time = min(task[task_No].finish_time_WR, min_vectorValue);
 
            if (min_vectorValue == minExec_Time)
            {
                auto it = find(ft_L.begin(), ft_L.end(), minExec_Time);
                int index;
                if (it != ft_L.end()) 
                {   
                    // calculating the index
                    index = it - ft_L.begin();
                }
                task[task_No].core_No = index+1;
                task[task_No].ready_time_L = minExec_Time - task[task_No].core_time[task[task_No].core_No]; 
                task[task_No].finish_time_L = minExec_Time;
                schedule[task[task_No].core_No] = minExec_Time;
                task[task_No].finish_time_WR = 0;
                task[task_No].finish_time_C = 0;
                task[task_No].finish_time_WS = 0;
                task[task_No].ready_time_WR = 0;
                task[task_No].ready_time_C = 0;
                task[task_No].ready_time_WS = 0;
            }
            else if (minExec_Time == task[task_No].finish_time_WR)
            {
                task[task_No].core_No = 0;
                task[task_No].is_cloud = 1;
                schedule[task[task_No].core_No] = minExec_Time;
                schedule_C = task[task_No].finish_time_C;
                schedule_WS = task[task_No].finish_time_WS;
                schedule_WR = task[task_No].finish_time_WR;
                task[task_No].finish_time_L = 0;
                task[task_No].ready_time_L = 0;
                
            }
        }
    }
}

void calculate_successor()
{
    int value = tasks_edge[0][0];
    int i, j = 0;
    
    //calculating successors of each task.
    for ( i = 0; i < n_e; i++)
    {
        task[tasks_edge[i][0]].successors.push_back(tasks_edge[i][1]);
    }
    cout << "*****************************************************************************************" << endl;
    cout << "\t SUCCESSORS" << endl;
    cout << "*****************************************************************************************" << endl;
    //printing successors of each task. 
    for (i = 1; i < task.size(); i ++)
    {
        cout << "   " << i << " successors are : ";  
        if(task[i].successors.size() == 0)
        {
            cout << "It is an exit task. so, no successors" << endl;
            task[i].is_exitTask = 1;
        }
        for (j = 0; j < task[i].successors.size(); j++)
        {
            cout << task[i].successors[j] << ", ";
        }
        cout << endl;
    }
}

void calculate_predecessors()
{
    int value = tasks_edge[0][1];
    int i, j = 0;
    
    //calculating predecessors of each task.
    for ( i = 0; i < n_e; i++)
    {
        task[tasks_edge[i][1]].predecessors.push_back(tasks_edge[i][0]);
    }
    cout << "*****************************************************************************************" << endl;
    cout << "\t PREDECESSORS" << endl;
    cout << "*****************************************************************************************" << endl;
    //printing predecessors of each task.
    for (i = 1; i < task.size(); i ++)
    {
        cout << "   " << i << " predecessors are : ";
        if(task[i].predecessors.size() == 0)
        {
            cout << "It is an initial task. so, no predecessors";
            task[i].is_initialTask = 1;
        }
        for (j = 0; j < task[i].predecessors.size(); j++)
        {
            cout << task[i].predecessors[j] << "  ";
        }
        cout << endl;
     }
}

int main()
{
    int i, j;
    cout << "Enter the number of cores in the device : ";
    cin >> n_c;
    cout << "Enter the number of tasks : ";
    cin >> n_t;
    cout << endl;
    
    task.resize(n_t+1);
    for (i = 1; i < task.size(); i++)
    {
        task[i].core_time.resize(n_c+1);
        task[i].task_number = i;
        cout << "Enter the task " << i << " completion time on each core : ";
        for (j = 1; j < task[i].core_time.size(); j++)
        {
            cin >> task[i].core_time[j];
        }
    }
    cout << endl;
    
    power.resize(n_c+1);
    schedule.resize(n_c+1);

    cout << "Enter the power consumption of cloud execution : ";
    cin >> power[0];
    
    for (i = 1; i < power.size(); i ++)
    {
        cout << "Enter the power consumption of each core " << i << " : ";
        cin >> power[i];
    }
    cout << endl;
    
    cout << "Enter the time for sending a task to communication channel : ";
    cin >> t_s;
    cout << "Enter the time to execute a task on cloud : ";
    cin >> t_c;
    cout << "Enter the time for sending back the result to device : ";
    cin >> t_r;
    cout << endl;
    T_re = t_s+t_c+t_r;
 
    cout << "Enter the number of edges in the task graph : ";
    cin >> n_e;
    cout << endl;
    
    cout << "Enter the parent task along with the task its pointed to" << endl;
    
    tasks_edge.resize(n_e);

    for (i = 0; i < n_e; i++)
    {
        tasks_edge[i].resize(2);
        for (j = 0; j < 2; j++)
        {
            cin >> tasks_edge[i][j];
        }
    }
    cout << endl;
    
    //calculating successors of each tasks
    calculate_successor();
    //calculating predecessors of each tasks
    calculate_predecessors();
    
    initial_scheduling();
    cout << "*****************************************************************************************" << endl;
    cout << "\t INITIAL SCHEDULING OUTPUT " << endl;
    print_taskVector(&task);
    
    //calculating completion time after initial scheduling
    total_Time = completion_Time(&task);
    T_max = 1.5 * total_Time;
    cout << "Time taken for task completion : " << total_Time << endl;
    
    //calculating enerygy energy_Consumption after initial scheduling
    initial_Energy = energy_Consumption(&task);
    cout << "Total energy consumed : " << initial_Energy << endl;
    
    //task migration 
    cout << endl;
    cout << "*****************************************************************************************" << endl;
    cout << "\t FINAL OUTPUT AFTER TASK MIGRATION " << endl;
    task_migration();
    
return 0;
}











