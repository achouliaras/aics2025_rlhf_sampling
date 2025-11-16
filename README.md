# aics2025_rlhf_sampling
Code for AICS2025 paper:  Maximizing the efficiency of human feedback in AI alignment: a comparative analysis

### Installation
```bash
git clone https://github.com/achouliaras/aics2025_rlhf_sampling.git
cd aics2025_rlhf_sampling
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```
### Run the experiments
```bash
python src/stochastic_ranking_search_space.py
python src/stochastic_ranking_search_space2.py
```
### Create the plots
```bash
python plot_win_tie_loss_probs.py
python plot_elo_probability.py
python plot_search_space.py
```
