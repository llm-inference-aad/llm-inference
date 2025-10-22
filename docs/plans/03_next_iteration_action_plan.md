# Next Iteration Action Plan

**Date:** October 21, 2025  
**Sprint Goal:** Validate fitness inheritance fix, implement server automation, and prepare optimization improvements.

---

## Phase 1: Critical Fixes & Infrastructure (MUST DO)

### 1.1 Server Shutdown Automation ⚡ HIGH PRIORITY
**Goal:** Prevent GPU waste and organize logs properly

**Tasks:**
- [ ] **Add server job tracking to `server.sh`**
  - Write `SLURM_JOB_ID` to `hostname_server_job.txt`
  - Location: After hostname write in `server.sh`

- [ ] **Implement automatic server shutdown in `run.sh`**
  - Add `scancel` logic after main job completes
  - Move server logs to `runs/{RUN_ID}/logs/`
  - Clean up tracking files

- [ ] **Increase server time limit safety net**
  - Change from 16 hours to 72 hours in `server.sh`

**Estimated Time:** 2 hours  
**Risk:** Low  
**Dependencies:** None

### 1.2 Enable Remote Fallback ⚡ HIGH PRIORITY  
**Goal:** Prevent evolution stall when primary server goes down

**Tasks:**
- [ ] **Update `.env` configuration**
  - Set `ENABLE_LLM_REMOTE_FALLBACK=true`
  - Verify `LLM_REMOTE_FALLBACK_TARGET=mixtral_hf`
  - Test HuggingFace credentials

**Estimated Time:** 30 minutes  
**Risk:** Low  
**Dependencies:** HuggingFace API access

### 1.3 Validation Run Setup ⚡ HIGH PRIORITY
**Goal:** Clean A/B test to measure fitness inheritance impact

**Tasks:**
- [ ] **Verify ancestry bug fix is in place**
  - Confirm line 344 in `run_improved.py` shows `'GENES':['network']`
  - Double-check no other ancestry bugs exist

- [ ] **Launch validation run with UNCHANGED parameters**
  - Keep all evolutionary constants identical to `auto_20251017_175557`
  - Same 8 generations, same population sizes
  - Document as fitness inheritance validation run

**Estimated Time:** 30 minutes setup + 16-24 hours run time  
**Risk:** Low  
**Dependencies:** Server automation (nice-to-have, not blocking)

---

## Phase 2: Monitoring & Analysis (RECOMMENDED)

### 2.1 Real-time Fitness Inheritance Monitoring 📊 MEDIUM PRIORITY
**Goal:** Confirm the bug fix is working during the run

**Tasks:**
- [ ] **Create monitoring script `scripts/monitor_inheritance.sh`**
  ```bash
  #!/bin/bash
  echo "Monitoring fitness inheritance events..."
  tail -f runs/latest/logs/slurm-main-*.out | grep --line-buffered "Inheriting fitness"
  ```

- [ ] **Set up ancestry verification check**
  - Script to check `checkpoint_gen_0.pkl` after Gen 0 completes
  - Verify Gen 0 individuals have `'GENES': ['network']`

**Estimated Time:** 1 hour  
**Risk:** Low  
**Dependencies:** Validation run in progress

### 2.2 Post-Run Analysis Scripts 📊 MEDIUM PRIORITY
**Goal:** Quantify the impact of fitness inheritance

**Tasks:**
- [ ] **Create `scripts/analyze_inheritance_impact.py`**
  - Count inheritance events
  - Calculate GPU hours saved
  - Compare goodput with previous run

- [ ] **Update plotting scripts**
  - Regenerate goodput analysis
  - Create Pareto front comparison
  - Document efficiency gains

**Estimated Time:** 2 hours  
**Risk:** Low  
**Dependencies:** Validation run complete

---

## Phase 3: Optimization Improvements (FUTURE WORK)

### 3.1 Retry Attempt Tracking 🔬 LOW PRIORITY
**Goal:** Data-driven optimization of retry logic

**Tasks:**
- [ ] **Modify `src/llm_mutation.py`**
  - Return retry count from mutation functions
  - Store `retry_attempts` in `GLOBAL_DATA`

- [ ] **Create `scripts/analyze_retry_effectiveness.py`**
  - Generate retry attempt distribution
  - Recommend optimal `LLM_GENERATION_MAX_RETRIES`

**Estimated Time:** 3 hours  
**Risk:** Medium (requires refactoring mutation functions)  
**Dependencies:** None (can be done in parallel)

### 3.2 Error-Aware Prompting (Simple Version) 🧠 LOW PRIORITY
**Goal:** Reduce fallback rate using error context

**Tasks:**
- [ ] **Analyze past `fallback_reason` patterns**
  - Extract common error types from previous runs
  - Create `COMMON_PITFALLS` guide

- [ ] **Append error guide to all prompts**
  - Modify template formatting in `src/llm_mutation.py`
  - A/B test impact on fallback rate

**Estimated Time:** 1 hour  
**Risk:** Low  
**Dependencies:** Historical fallback data analysis

---

## Phase 4: Extended Experiments (LATER ITERATIONS)

### 4.1 Convergence Analysis 📈 FUTURE
**Goal:** Determine optimal run length

**Tasks:**
- [ ] **15-generation run after validation**
- [ ] **Pareto front stabilization analysis**
- [ ] **Compute cost vs. solution quality trade-offs**

### 4.2 Population Diversity Experiments 🔬 FUTURE  
**Goal:** Explore larger population benefits

**Tasks:**
- [ ] **Test `population_size=12` configuration**
- [ ] **Measure genotype/phenotype diversity metrics**
- [ ] **Cost-benefit analysis of larger populations**

---

## Success Criteria & Metrics

### Must-Have Success (Phase 1)
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Fitness Inheritance Events** | ≥5 events | `grep -c "Inheriting fitness" logs/slurm-main-*.out` |
| **Server Automated Shutdown** | 100% success | Server logs moved to run directory |
| **No Server Timeouts** | 0 timeouts | Check for connection errors in logs |
| **Goodput Maintained** | ≥95% | `scripts/plot_latency_vs_goodput.py` |

### Nice-to-Have Success (Phase 2)
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **GPU Hours Saved** | ~8 hours | `(inheritance_count * avg_eval_time)` |
| **Real-time Monitoring** | Working dashboard | Monitor script shows inheritance events |
| **Analysis Complete** | Full report | Impact analysis document generated |

---

## Implementation Timeline

### Week 1 (Current)
- **Day 1:** Phase 1.1-1.2 (Server automation + remote fallback) ⚡
- **Day 2:** Phase 1.3 + 2.1 (Launch validation run + monitoring) ⚡  
- **Day 3-4:** Run in progress, Phase 3.1 development in parallel 📊
- **Day 5:** Phase 2.2 (Post-run analysis) 📊

### Week 2 (If time permits)
- **Phase 3.2:** Error-aware prompting implementation
- **Phase 4 planning:** Extended experiments design

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation Plan |
|------|------------|--------|----------------|
| **Fitness inheritance still broken** | Low | High | Monitor ancestry structure in Gen 0 checkpoint |
| **Server automation bugs** | Medium | Medium | Test on small dummy job first |
| **Remote fallback failures** | Medium | High | Verify HuggingFace credentials before launch |
| **Analysis scripts crash** | Low | Low | Use try-catch blocks and graceful degradation |

---

## Decision Points

**CONFIRM BEFORE PROCEEDING:**

1. **Phase 1 Priority:** Do we implement server automation before launching the validation run, or launch immediately?
   - **Recommendation:** Do server automation first (2 hours) for cleaner logs

2. **Monitoring Level:** How much real-time monitoring do we want during the run?
   - **Recommendation:** Basic inheritance event monitoring only

3. **Parallel Development:** Should we work on Phase 3 improvements while validation run is in progress?
   - **Recommendation:** Yes, retry tracking can be developed in parallel

4. **Documentation:** How detailed should the impact analysis report be?
   - **Recommendation:** Focus on key metrics (GPU hours saved, inheritance events) for SCRUM update

---

## Next Steps

**Ready for your confirmation:**

1. ✅ **Approve Phase 1** (Critical fixes - server automation, remote fallback, validation run)
2. ✅ **Approve Phase 2** (Monitoring and analysis)  
3. 🤔 **Decide on Phase 3** (Optimization improvements - do now or later?)
4. 📋 **Confirm timeline** (1 week for Phases 1-2, or extend?)

**Once confirmed, I can start implementing in this order:**
1. Server automation (`server.sh` + `run.sh` modifications)
2. Remote fallback (`.env` configuration)  
3. Validation run setup and launch
4. Monitoring scripts while run is in progress