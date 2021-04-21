#!/usr/bin/env bash

## This script installs AWS CLI __Version 2__
##
## Do not forget to add the path to PATH
##     export PATH=$PATH:$HOME/local/bin

VERSION="2.1.38"
ZIP_FILE="awscliv2.zip"

cd /tmp
wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-${VERSION}.zip" -O "${ZIP_FILE}"
unzip ${ZIP_FILE}
./aws/install -i ${HOME}/local/aws-cli -b ${HOME}/local/bin
rm -f ${ZIP_FILE}
rm -fr aws
~/local/bin