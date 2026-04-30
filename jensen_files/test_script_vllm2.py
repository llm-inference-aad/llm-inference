import requests
import threading
import time
import statistics

lock = threading.Lock()

NUM_REQUESTS = 8      # concurrent requests per round
NUM_ROUNDS = 10       # number of rounds


def send_request(idx, timings):
    """Send a request to the vLLM OpenAI-compatible API and record latency in `timings`."""
    url = "http://atl1-1-03-010-10-0.pace.gatech.edu:8000/v1/completions"

    prompts = [
        """Code Snippet 1
```python
import torch.nn as nn
import torch.nn.functional as F

class MinPool2d_x(nn.Module):
    def __init__(self, ks, ceil_mode):
        super().__init__()
        self.ks = ks
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return -F.max_pool2d(-x, self.ks, ceil_mode=self.ceil_mode)

class EVE(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.minpool = MinPool2d_x(2, ceil_mode=True)
        self.pw = nn.Sequential(
            nn.Conv2d(2*cin, cout, 1, 1, bias=False),
            nn.BatchNorm2d(cout)
        )

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x)
        ), 1)
        x = self.pw(x)
        return x
```
Code Snippet 2
```python
class MinPool2d_x(nn.Module):
    def __init__(self, ks, ceil_mode):
        super().__init__()
        self.ks = ks
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return -F.max_pool2d(-x, self.ks, ceil_mode=self.ceil_mode)

class EVE(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.minpool = MinPool2d_x(2, ceil_mode=True)
        self.pw = nn.Conv2d(2*cin, cout, 1, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x)
        ), 1)
        x = self.pw(x)
        x = self.bn_pw(x)
        x = self.act(x)
        return x
```

Q: How can the model's predictive metrics be enhanced by amalgamating elements from these two code snippet alternatives?

A: Let us think step by step""",
        """Code Snippet 1
```python
import torch.nn as nn

class SE_LN_Improved(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))
        self.ln = nn.LayerNorm(cin)
        self.bn = nn.BatchNorm1d(cin)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1, x.size(1))
        x = self.ln(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.act2(x)
        x = x.view(-1, x.size(1), 1, 1)
        return x * y
```
Code Snippet 2
```python
class SE(nn.Module):
    def __init__(self, cin, ratio):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(cin, int(cin/ratio), bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1.out_features, cin, bias=False)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1,x.size()[1])
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = x.view(-1,x.size()[1],1,1)
        return x*y
```

Q: How can the model's performance or efficiency be elevated by amalgamating elements from these two code snippet alternatives?

A: Let us think step by step""",
        """Code Snippet 1
```python
def get_optimizer(model, lr, weight_decay=0, nesterov=True):
    if weight_decay != 0:
        g0, g1, g2 = [], [], []
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                g2.append(v.bias)
            if isinstance(v, (nn.BatchNorm2d, nn.LayerNorm)):
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g1.append(v.weight)

        opt = SGD(g0+g1, lr, 0.9, nesterov=nesterov)
        opt.add_param_group({'params': g2})
    else:
        opt = SGD(model.parameters(), lr, 0.9, nesterov=nesterov)
    return opt
```
Code Snippet 2
```python
def get_optimizer(model, lr, weight_decay=0, nesterov=True):
    if weight_decay != 0:
        g0, g1, g2 = [], [], []
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                g2.append(v.bias)
            if isinstance(v, (nn.BatchNorm2d, nn.LayerNorm)):
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g1.append(v.weight)

        opt = SGD(g0, lr, 0.9, nesterov=nesterov)
        opt.add_param_group({'params': g1, 'weight_decay': weight_decay})
        opt.add_param_group({'params': g2})
    else:
        opt = SGD(model.parameters(), lr, 0.9, nesterov=nesterov)
    return opt
```

Q: How can the model's predictive metrics be enhanced by amalgamating elements from these two code snippet alternatives?

A: Let us think step by step""",
        """Code Snippet 1
```python
import torch.nn as nn

def pad_num_y(k_s):
    pad_per_side = int((k_s-1)*0.5)
    return pad_per_side

class DW(nn.Module):
    def __init__(self, cin, dw_s):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, dw_s, 1, pad_num_y(dw_s), groups=cin)
        self.act = nn.Hardswish()
        self.dw_small = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)

    def forward(self, x):
        dw_x = self.dw(x)
        dw_x = self.act(dw_x)

        dw_small_x = self.dw_small(x)
        dw_small_x = self.act(dw_small_x)

        x = dw_x + dw_small_x
        return x
```
Code Snippet 2
```python
def pad_num_y(k_s):
    pad_per_side = int((k_s-1)*0.5)
    return pad_per_side

class DW(nn.Module):
    def __init__(self, cin, dw_s):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, dw_s, 1, pad_num_y(dw_s), groups=cin)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.dw(x)
        x = self.act(x)
        return x
```

Q: How can the model's predictive metrics be enhanced by amalgamating elements from these two code snippet alternatives?

A: Let us think step by step"""
    ]

    payload = {
        "model": "/home/hice1/jzhang3318/scratch/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct",
        "prompt": prompts[idx % len(prompts)],
        "max_tokens": 3000,
        "temperature": 0.2,
    }

    headers = {"Content-Type": "application/json"}

    with lock:
        print(f"Sending request {idx} at {time.strftime('%H:%M:%S', time.localtime())}")

    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload)
        elapsed = time.time() - start_time
    except Exception as e:
        elapsed = time.time() - start_time
        with lock:
            print(f"Request {idx} failed with exception: {e}")
        with lock:
            timings.append(elapsed)
        return

    with lock:
        timings.append(elapsed)

    if response.status_code == 200:
        try:
            data = response.json()
            text = data.get("choices", [{}])[0].get("text", "")
        except ValueError:
            text = ""
        with lock:
            print(f"Response {idx} received in {elapsed:.2f}s (len={len(text)} chars)\n")
    else:
        with lock:
            print(f"Error in request {idx}: {response.status_code}")
            print(response.text)


def main():
    all_request_times = []
    round_durations = []

    for round_idx in range(NUM_ROUNDS):
        with lock:
            print(f"\n=== Round {round_idx + 1}/{NUM_ROUNDS} ===")

        round_start = time.time()
        round_timings = []

        threads = []
        for i in range(NUM_REQUESTS):
            global_idx = round_idx * NUM_REQUESTS + i
            t = threading.Thread(target=send_request, args=(global_idx, round_timings))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        round_end = time.time()
        duration = round_end - round_start
        round_durations.append(duration)
        all_request_times.extend(round_timings)

        with lock:
            print(f"Round {round_idx + 1} completed in {duration:.2f}s")
            if round_timings:
                print(f"  Avg request time this round: {statistics.mean(round_timings):.2f}s")
                print(f"  Min/Max this round: {min(round_timings):.2f}s / {max(round_timings):.2f}s")

    print("\nAll rounds completed!")

    total_requests = len(all_request_times)
    if total_requests == 0:
        print("No timings recorded.")
        return

    avg_time = statistics.mean(all_request_times)
    median_time = statistics.median(all_request_times)
    min_time = min(all_request_times)
    max_time = max(all_request_times)
    stdev_time = statistics.pstdev(all_request_times) if total_requests > 1 else 0.0

    avg_round = statistics.mean(round_durations)
    median_round = statistics.median(round_durations)

    print("\n=== Overall Statistics ===")
    print(f"Total requests: {total_requests}")
    print(f"Average time per request: {avg_time:.4f}s")
    print(f"Median time per request: {median_time:.4f}s")
    print(f"Min/Max time per request: {min_time:.4f}s / {max_time:.4f}s")
    print(f"Std dev time per request: {stdev_time:.4f}s")

    print(f"\nRounds: {NUM_ROUNDS}, Requests per round: {NUM_REQUESTS}")
    print(f"Average round duration: {avg_round:.4f}s")
    print(f"Median round duration: {median_round:.4f}s")
    print(f"Total wall-clock time (sum of rounds): {sum(round_durations):.4f}s")
    print(f"Total time / total requests: {(sum(round_durations) / total_requests):.4f}s")


if __name__ == "__main__":
    main()
