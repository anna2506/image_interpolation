This code is simulate digital zoom in a camera.</br>
1)To run it first you should download Miniconda on your PC:
### Linux
###### $ wget https :// repo . anaconda . com / miniconda / Miniconda3 - latest -Linux - x86_64 .sh
### MacOSX
###### $ curl -O https :// repo . anaconda . com / miniconda / Miniconda3 - latest - MacOSX - x86_64 .sh
### Windows
For Windows, download the installer from https://repo.anaconda.com/miniconda/Miniconda3-lates
t-Windows-x86_64.exe
2)Install Miniconda into your home directory:
### Linux
###### $ sh Miniconda3 - latest -Linux - x86_64 .sh -b -p $HOME / miniconda
### MacOSX
###### $ sh Miniconda3 - latest -Linux - x86_64 .sh
### Windows
For Windows, execute the installer and follow the steps of the installation wizard</br>
3)To start using Miniconda, you need to make sure that it is in your $PATH. For example, on Linux (or
MacOSX), you can simply execute:</br>
###### $ export PATH = $HOME / miniconda / bin: $PATH</br>
You can also add it to your home .bashrc (or .bash_profile on MacOSX), so that you can skip this
step the next time you start the terminal session.</br>
4)To test if the installation was successful, command conda list should return the list of installed packages.</br>
5)Then we create a new conda environment with the name you want(e.x. "cv1"). We assume that the file requirements.txt
with the dependencies we provide is in the current directory, so simply execute:</br>
###### $ conda env create -f= requirements . txt -n cv1</br>
###### $ source activate cv1</br>
The second command activates our environment cv1. Now you can run this Python code. Enjoy!</br>
You can try every image you want in the path "data/image.png".</br>
This code first put the bayer filter to your image, then it makes scaling and cropping it(x2).</br>
After this makes nearest neighbour interpolation for RED and BLUE color and bilinear interpolation for GREEN colors.
