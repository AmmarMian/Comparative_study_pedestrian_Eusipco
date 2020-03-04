**********************************
# A Comparative Study of Supervised Learning Algorithms for Symmetric Positive Definite Features

This repository contains code for the paper "A Comparative Study of Supervised Learning Algorithms for Symmetric Positive Definite Features” Submitted to EUSIPCO 2020.

It allows to consider Riemannian classification algorithms on the task of pedestrian detection.

----

## Requirements

The code is made in Python 3.7 with an Anaconda distribution. The pyriemann package (https://github.com/alexandrebarachant/pyRiemann) is used and has been copied to this repository.

---

## Datasets

Two set of data can be used: The INRAI dataset available at http://pascal.inrialpes.fr/data/human/ and the DaimerChrysler one available at http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Daimler_Mono_Ped__Class__Bench/daimler_mono_ped__class__bench.html.

After downloading the datasets, extract them to some folder and update the paths in paths.yaml. The path should be relative to base of this repository, which is the folder of this README.md.

**********************************
## Executing the simulations

* First download the datsets and update paths.yaml as needed.
* Execute the files job_INRIA.sh, job_DaimerChrysler.sh to obtain the results presented in tables I and II of the paper. The results will be available  in the folder: pedestrian_detection/Simulation_data/Simulations_setups_data.
* In order to modify the parameters of the simulation, it is only needed to change the setup files in Simulation_setups/ folder.
* For the robust study,  execute  job_robust_study.sh which use pre-computed covariance features data in the folder pedestrian_detection/Scripts/Tuning/Data. These features are obtained using the simulation_setups orignally in this folder.

**********************************
## Authors

The folder was created by:

* Ammar Mian, Postdoctoral researcher at Aalto University in the Department of Signal Processing and Acoustics.
  Contact: ammar.mian@aalto.fi
  Web: https://ammarmian.github.io/
* Elias Raninen, Ph.D Student at Aalto University in the Department of Signal Processing and Acoustics.
* Esa Ollila, Professor at Aalto University in the Department of Signal Processing and Acoustics

**********************************
## Miscellaneous

THIS CODE IS PROVIDED “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PAGERDUTY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) SUSTAINED BY YOU OR A THIRD PARTY, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT ARISING IN ANY WAY OUT OF THE USE OF THIS SAMPLE CODE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
