add wave -position insertpoint  \
sim/:tb_NR_signal_generator:A \
sim/:tb_NR_signal_generator:initdone \
sim/:tb_NR_signal_generator:clock \
sim/:tb_NR_signal_generator:Z \

run -all
