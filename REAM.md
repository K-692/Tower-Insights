# 🗼 Tower Insights: Automated Telecom Tower Inspection & Visual Analytics

> Note: This file (`REAM.md`) is provided as an exact alias for [`README.md`](README.md). Please refer to [`README.md`](README.md) for full project documentation, architecture diagrams, and quickstart instructions.

---

## 📌 Quick Summary

**Tower Insights** is an end-to-end deep learning and computer vision platform designed for telecommunications tower asset management:
- **Monocular Height Estimation:** Derives real-world metric height using anchor base width scaling ($S = W_{\text{base}} / w_{\text{adj}}$) and ground elevation foreshortening angle compensation ($H_{\text{final}} = 1.8 \times H_{\text{raw}}$).
- **Construction Milestone Verification:** Evaluates progress against design targets ($30\text{m}$, $40\text{m}$, $60\text{m}$) to classify structures as Completed ($\ge 85\%$) or Under Construction.
- **Machinery Inventory Detection:** Detects and counts Base Transceiver Stations (BTS), Control Boxes, Generators, and Solar Panels.
- **Antenna Array Auditing:** Identifies and counts GSM panel antennas and Microwave dish backhauls.
- **Interactive Web Dashboard:** Built with Streamlit for zero-friction in-field operational auditing.

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/K-692/Tower-Insights.git
cd Tower-Insights

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run main.py
```

For complete documentation, mathematical formulations, and research blueprint, see:
- [`README.md`](README.md) - Project overview and user guide.
- [`flow.md`](flow.md) - Detailed implementation journey and research paper documentation.
