# Designs for Testing and Evaluation

This directory contains the designs for testing out and running experiments with SylQ-SV. In `test-designs/`
there are some smaller designs for convenience and testing purposes. The larger designs we used in our evaluation can be cloned and accessed below:

* [Buggy OR1200 CPU ](https://github.com/HWSec-UNC/buggy-or1200/tree/d11093d51fb127fa9f8d032143da049220031522)
* [HACK@DAC 2018 ](https://github.com/HWSec-UNC/verification-benchmarks/tree/main/hackatdac18)
* [HACK@DAC 2019 ](https://github.com/HWSec-UNC/verification-benchmarks/tree/main/hackatdac19)
* [HACK@DAC 2021 ](https://github.com/HWSec-UNC/verification-benchmarks/tree/main/hackatdac21)
* [OpenTitan Ibex Core ](https://github.com/lowRISC/ibex.git)

The git submodules in this directory also point to the commits used for evaluation of SylQ in our paper.
The properties for the OR1200 and HACK@DAC designs are accessible in the benchmarks repository linked above. The properties for each design are also located at the URL provided for each design above. If you use the properties, please cite our paper below: 

@inproceedings{rogers2025properties,
  title={Hardware Security Benchmarks for Open-Source SystemVerilog
  Designs},
  author={Jayden Rogers and Niyaz Shakeel and Divya Mankani and Samantha Espinosa and Cade Chabra and Kaki Ryan and Cynthia Sturton},
  booktitle={SecDev},
  year={2025},
  organization={IEEE}
}
