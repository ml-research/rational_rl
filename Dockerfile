FROM nvidia/cuda:10.1-devel
ARG 	username=rl_paus
RUN 	apt-get update -y && \
	apt-get install -y apt-utils git curl sudo nano vim python3-pyqt5 python3-pip

RUN	useradd -m $username && echo $username":test" | chpasswd && adduser $username sudo


COPY 	./requirements.txt /home/$username

WORKDIR /home/$username/
RUN 	pip3 install --upgrade pip
RUN 	pip3 install -r requirements.txt
RUN	pip3 install mushroom_rl[all]
RUN	git clone https://github.com/alejandromolinaml/activation_functions.git /pau_install && \
	cd /pau_install/src && \
	pip3 install -r requirements.txt
# CMD	cd /pau_install/src
# CMD	python3 setup.py install
