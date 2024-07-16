#!/usr/bin/bash
# This script must be executed out of {project_root}/data and made executabale $ chmod +x cleandata.sh
sed 's/ //g' census.csv >  clean_census.csv