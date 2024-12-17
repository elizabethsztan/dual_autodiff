# Docker Environment for Testing `dual_autodiff` Packages

This project provides a Dockerized environment for testing the `dual_autodiff` and `dual_autodiff_x` Python packages. The environment includes Jupyter Notebook for interactive testing.

---

## Prerequisites

1. [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running on your system.
2. `dual_autodiff` and `dual_autodiff_x` `.whl` files located in the `wheels/` directory. The wheel using python3.10 is used in this testing file. 
3. Jupyter notebooks for testing located in the `notebooks/` directory.
4. In the root directory ('Docker_testing') run the following command in terminal
```bash
docker build -t test-dual-autodiff .
```
5. Run the container
```bash
docker run -p 8888:8888 test-dual-autodiff
```
6. You will see a URL like 
```bash
http://127.0.0.1:8888/?token=<token>
```
copy and paste this into your browser. The jupypter notebooks will be available which you can run.

7. You can stop running the container from Docker Desktop.
---

