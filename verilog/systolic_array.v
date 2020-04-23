///////////////////////////////////////
//
//
// this is the subarray module
module j_systolic_array #(
    parameter SUBARRAY_WIDTH     = 8 , // width of the subarray
    parameter SUBARRAY_HEIGHT    = 8 , // height of the subarray
    parameter NUM_COE_ARRAY      = 16,
    parameter NUM_COMBINED_TERMS = 8 ,
    parameter NUM_BIT_EXPONENT   = 4
) (
    input                                                           clk            ,
    input                                                           reset          ,
    input  [   3:0] data_terms,
    input  [   4:0] group_size,
    input  [   6:0] group_budget,
    input  [                   8*NUM_COE_ARRAY*SUBARRAY_HEIGHT-1:0] accumulation_in,
    input  [NUM_COMBINED_TERMS*NUM_BIT_EXPONENT*SUBARRAY_WIDTH-1:0] dataflow_in    ,
    input  [                 NUM_COMBINED_TERMS*SUBARRAY_WIDTH-1:0] sign_flow_in   ,
    output [                   8*NUM_COE_ARRAY*SUBARRAY_HEIGHT-1:0] result
);


    function [31:0] clog2 (input [31:0] x);
        reg [31:0] x_tmp;
        begin
            x_tmp = x-1;
            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                x_tmp = x_tmp >> 1;
            end
        end
    endfunction


// create the systolic array
    wire [NUM_COMBINED_TERMS*NUM_BIT_EXPONENT-1:0] arr_dataflow_in    [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [NUM_COMBINED_TERMS*NUM_BIT_EXPONENT-1:0] arr_dataflow_out   [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [                 NUM_COMBINED_TERMS-1:0] arr_sign_inputs_in [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [                 NUM_COMBINED_TERMS-1:0] arr_sign_inputs_out[SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [                    8*NUM_COE_ARRAY-1:0] arr_accumulation_in[SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [                    8*NUM_COE_ARRAY-1:0] arr_result         [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];

    genvar i,j;
    generate
        for (j=0; j<SUBARRAY_HEIGHT; j=j+1) begin: g_H
            for (i=0; i<SUBARRAY_WIDTH; i=i+1) begin: g_W

                if(j==0) begin: g_j_eq_0
                    assign arr_dataflow_in[i][0] = dataflow_in    [i*4*NUM_BIT_EXPONENT +: 4*NUM_BIT_EXPONENT];
                    assign arr_sign_inputs_in[i][0] = sign_flow_in   [i*4 +: 4];
                end else begin: g_j_others
                    assign arr_dataflow_in[i][j]    = arr_dataflow_out       [i][j-1];
                    assign arr_sign_inputs_in[i][j] = arr_sign_inputs_out    [i][j-1];
                end

                if(i==0) begin: g_i_eq_0
                    assign arr_accumulation_in[0][j] = accumulation_in  [2*NUM_COE_ARRAY*j +: 2*NUM_COE_ARRAY];
                end else begin: g_i_others
                    assign arr_accumulation_in[i][j] = arr_result             [i-1][j];
                end

                if(i==(SUBARRAY_WIDTH-1)) begin: g_i_eq_W
                    assign result[2*NUM_COE_ARRAY*j+:2*NUM_COE_ARRAY] = arr_result       [SUBARRAY_WIDTH-1][j];
                end

                mx_cell #(.NUM_BIT_EXPONENT(NUM_BIT_EXPONENT),.NUM_COE_ARRAY(NUM_COE_ARRAY),.NUM_COMBINED_TERMS(NUM_COMBINED_TERMS))
                    i_j_MX_cell (
                        .clk               (clk                                                             ),
                        .sign_in           (arr_sign_inputs_in  [i][j]                                      ),
                        .sign_out          (arr_sign_inputs_out [i][j]                                      ),
                        .reset             (reset                                                           ),
                        .data_terms        (data_terms                                                      ),
                        .group_size        (group_size                                                      ),
                        .group_budget      (group_budget                                                    ),
                        .dataflow_in       (arr_dataflow_in       [i][j]                                    ),
                        .dataflow_out      (arr_dataflow_out      [i][j]                                    ),
                        .accumulation_in   (arr_accumulation_in   [i][j]                                    ),
                        .result            (arr_result            [i][j]                                    )
                    );

            end
        end
    endgenerate
endmodule






