# DARPCD INSTANCES DESCRIPTION

-----------------
# DataX_a.txt files

HEADER

0 column: no of requests

1 column: vehicle capacity

2 column: earliest possible start time in hours (to be multiplied by 60)

3 column: latest possible end time in hours (to be multiplied by 60)

4 column: vehicle speed

5 column: fixed time for unloading and reloading at the depot

6 column: fixed time required for uploading and loading a customer

7 column: number of vehicles

8 column: max route duration

9 column: maximum allowed ride time

10 column: in-vehicle loading point after which the perception of travel times is affected 

11 column: in-vehicle travel time penalty due to overcrowding

BODY

0 column: request if value > 0 and depot if value = 0

1 column: latitude of the destination of the request

2 column: longitude of the destination of the request

3 column: latitude of the origin of the request

4 column: longitude of the origin of the request

5 column: passenger demand for this request

-----------------

# DataX_a_tw.txt files

0 column: earliest time to serve vertex n+i, where i is the row number start counting from 1

1 column: latest time to serve vertex n+i, where i is the row number start counting from 1

2 column: earliest time to serve vertex i, where i is the row number start counting from 1

3 column: latest time to serve vertex i, where i is the row number start counting from 1
