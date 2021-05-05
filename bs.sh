if [[ ${NUM: -1} == "M" ]]; then
		echo "M"
elif [[ "${NUM: -1}" == "K" ]]; then
    echo "K"
else echo "Error only M or K is allowed"
fi