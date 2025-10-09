import os 
import GPUtil
import time
def get_gpu_info():
   gpulist = []
   Gpus = GPUtil.getGPUs()
   GPUtil.showUtilization()
   for gpu in Gpus:
      # modify the rules, default 5%.
      if gpu.load<0.05 and gpu.memoryUtil<0.05:
         gpulist.append(gpu.id)
   return gpulist
import json
def read_json_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
def main():
   data=read_json_file('todo.json')
   for i in data:
      while True:
         # detect the usage of GPUs
         gpu_info = get_gpu_info()
         if gpu_info != [] and len(gpu_info)>= i['gpu_require']:
            f= lambda gpu_info,i: ','.join(map(str, gpu_info[:i['gpu_require']]))
            gpu_use=f(gpu_info,i)
            print("Task starts")
            # write your own command here,and its available to design a personal json file if you need other args.
            os.system(f"deepspeed --include localhost:{gpu_use} --master_port 29505 {i['task']} > {i['log']}")
            break
         else:
            time.sleep(10)
if __name__ == '__main__':
   main()
