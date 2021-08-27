FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
ARG	folder_name=rational_rl
RUN 	apt-get update -y && apt-get install -y python3-pyqt5 python3-pip
RUN	mkdir /home/$folder_name

COPY	./requirements.txt /home/$folder_name

WORKDIR /home/$folder_name/
RUN	pip3 install --upgrade pip
RUN	pip3 install -r requirements.txt
RUN	apt-get install -y swig
RUN	pip3 install mushroom_rl[all]
