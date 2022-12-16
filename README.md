# octree_cloud_merge

This is the code companion for the following paper:

**[Paper] [Digital Outcrop Model Generation from Hybrid UAV and Panoramic Imaging Systems](http://dx.doi.org/10.3390/rs14163994)**

http://dx.doi.org/10.3390/rs14163994


Authors:
  [Alysson Soares Aires<sup>1</sup>](https://www.researchgate.net/profile/Alysson-Soares-Aires),
  [Ademir Marques Junior<sup>1</sup>](https://www.researchgate.net/profile/Ademir_Junior),
  [Daniel Zanotta<sup>1</sup>](https://www.researchgate.net/profile/Daniel_Zanotta),
  [André Luis Spigolon<sup>2</sup>](https://www.researchgate.net/profile/Ald-Spigolon),
  [Maurício Roberto Veronez<sup>1</sup>](https://www.researchgate.net/profile/Mauricio_Veronez),
  [Luiz Gonzaga Junior<sup>1</sup>](https://www.researchgate.net/profile/Luiz_Gonzaga_da_Silveira_Jr)
 
<sup>1</sup>[Vizlab | X-Reality and GeoInformatics Lab<sup>1</sup>](http://vizlab.unisinos.br/), 
<sup>2</sup>[CENPES-Petrobras](https://petrobras.com.br/en/our-activities/technology-innovation/)  

The study of outcrops in Geosciences is being significantly improved by the enhancement of technologies that aims to build Digital Outcrop Models (DOMs). Usually, the virtual environment is built by a collection of partially overlapped photographs taken from diverse perspectives, frequently using unmanned aerial vehicles (UAV). However, in situations including very steep features or even sub-vertical patterns, it is expected an incomplete covering of objects. This work proposes an integration framework between terrestrial Spherical Panoramic Images (SPI), acquired by omnidirectional fusion camera, and UAV survey to overcome gaps left by traditional mapping in complex natural structures like outcrops. The omnidirectional fusion camera produces wider field of view images from different perspectives able to considerably improve the representation of the DOM, mainly where UAV has geometric view restrictions. We designed controlled experiments to guaranty the equivalent performance of SPI compared with UAV. The adaptive integration is made through an optimized selective strategy based on an octree framework. The quality of the 3D model generated using this approach was assessed by quantitative and qualitative indicators. The results show the potential of generating a more reliable 3D model using SPI allied with UAV image data while reducing field survey time and complexity.


<img src="https://github.com/ademirmarquesjunior/octree_cloud_merge/blob/main/octree_structure.png" width="500" alt="Segmented image">


## Requirements

This Python script requires a Python 3 environment and the following installed libraries as seen in the file requirements.txt.

```bash
Open3D
Scipy
pyntcloud
sklearn
pandas
numpy
```



## Installation and Usage

Download this repository and run the "pqa.py" file in a Python 3 environment.



## Credits	
This work is credited to the [Vizlab | X-Reality and GeoInformatics Lab](http://vizlab.unisinos.br/) and the following developers:	[Ademir Marques Junior](https://www.researchgate.net/profile/Ademir_Junior), [Alysson Soares Aires](https://www.researchgate.net/profile/Alysson-Soares-Aires) e [Daniel Zanotta](https://www.researchgate.net/profile/Daniel_Zanotta).

## License

    MIT Licence

## How to cite

If you find our work useful in your research please consider citing our papers:

```bash
Aires, Alysson Soares, Ademir Marques Junior, Daniel Capella Zanotta, André Luiz Durante Spigolon, Mauricio Roberto Veronez, and Luiz Gonzaga Jr. 2022.
"Digital Outcrop Model Generation from Hybrid UAV and Panoramic Imaging Systems" Remote Sensing 14, no. 16: 3994. https://doi.org/10.3390/rs14163994
```
