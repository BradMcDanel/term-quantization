module systolic_dla_top #(
    parameter SUBARRAY_WIDTH     = 64  , // width of the subarray
    parameter SUBARRAY_HEIGHT    = 128 , // height of the subarray
    parameter NUM_COE_ARRAY      = 16,
    parameter NUM_COMBINED_TERMS = 8 ,
    parameter NUM_BIT_EXPONENT   = 3
) (
    input                                                           clk            ,
    input                                                           reset          ,
    input  [                                                   1:0] turn_on        ,
    input                                                           psel           ,
    input  [                                                  15:0] paddr          ,
    input                                                           pwrite         ,
    input  [                                                  31:0] pwdata         ,
    input                                                           penable        ,
    output [                                                  31:0] prdata         ,
    output                                                          pready         ,
    input  [                   8*NUM_COE_ARRAY*SUBARRAY_HEIGHT-1:0] accumulation_in,
    input  [NUM_COMBINED_TERMS*NUM_BIT_EXPONENT*SUBARRAY_WIDTH-1:0] dataflow_in    ,
    input  [                 NUM_COMBINED_TERMS*SUBARRAY_WIDTH-1:0] sign_flow_in   ,
    output [                   8*NUM_COE_ARRAY*SUBARRAY_HEIGHT-1:0] result         ,
    output [                   8*NUM_COE_ARRAY*SUBARRAY_HEIGHT-1:0] result_sign
);

    wire [1:0]    turn_on_signal;
    wire [4:0]    group_size    ;
    wire [6:0]    group_budget  ;
    wire [3:0]    data_terms    ;

    wire [31:0] reg_write_data;
    wire [15:0] reg_addr      ;
    wire [31:0] reg_read_data ;
    wire        reg_write     ;
    wire        reg_read      ;
    wire        reg_idle      ;

    apb2reg i_apb2reg (
        .clk           (clk           ),
        .reset_n       (~reset        ),
        .psel          (psel          ),
        .paddr         (paddr[15:2]   ),
        .pwrite        (pwrite        ),
        .pwdata        (pwdata        ),
        .penable       (penable       ),
        .prdata        (prdata        ),
        .pready        (pready        ),
        .reg_write_data(reg_write_data),
        .reg_addr      (reg_addr      ),
        .reg_read_data (reg_read_data ),
        .reg_write     (reg_write     ),
        .reg_read      (reg_read      ),
        .reg_idle      (reg_idle      )
    );


    reg_define i_reg_define (
        .turn_on_signal(turn_on_signal                      ),
        .input_acc_size({data_terms,group_budget,group_size}),
        .write_data    (reg_write_data                      ),
        .addr          (reg_addr                            ),
        .read_data     (reg_read_data                       ),
        .write         (reg_write                           ),
        .read          (reg_read                            ),
        .clk           (clk                                 )
    );


    //////////////////////////////////////////////////////
    // Systolic main
    //////////////////////////////////////////////////////

    wire [8*NUM_COE_ARRAY*SUBARRAY_HEIGHT-1:0] systolic_result;

    j_systolic_array #(
        .SUBARRAY_WIDTH    (SUBARRAY_WIDTH    ),
        .SUBARRAY_HEIGHT   (SUBARRAY_HEIGHT   ),
        .NUM_COE_ARRAY     (NUM_COE_ARRAY     ),
        .NUM_COMBINED_TERMS(NUM_COMBINED_TERMS),
        .NUM_BIT_EXPONENT  (NUM_BIT_EXPONENT  )
    ) i_j_systolic_array (
        .clk            (clk            ),
        .reset          (reset          ),
        .dataflow_in    (dataflow_in    ),
        .sign_flow_in   (sign_flow_in   ),
        .data_terms     (data_terms     ),
        .group_size     (group_size     ),
        .group_budget   (group_budget   ),
        .accumulation_in(accumulation_in),
        .result         (systolic_result)
    );

    ////////////////////////////////////////////////////////////
    // Binary Stream Converter
    ////////////////////////////////////////////////////////////
    wire [8*SUBARRAY_HEIGHT-1:0] converter_output;
    genvar k;
    generate
        for (k = 0; k < 8*SUBARRAY_HEIGHT; k=k+1) begin: j_converter
            binary_stream_converter #(.NUM_COE_ARRAY(NUM_COE_ARRAY),.NUM_COMBINED_TERMS(NUM_COMBINED_TERMS)) j_binary_stream_converter (
                .clk          (clk                                                              ),
                .reset      (reset                                                          ),
                .systolic_result  (systolic_result[NUM_COMBINED_TERMS*(k+1)-1:NUM_COMBINED_TERMS*k]),
                .binary_result (converter_output [k])
            );

        end

    endgenerate

    ////////////////////////////////////////////////////////////
    // ReLU and requantizer
    ////////////////////////////////////////////////////////////
    genvar i;
    wire [8*SUBARRAY_HEIGHT-1:0] requant_serial_output;
    generate
        for (i = 0; i < 8*SUBARRAY_HEIGHT; i=i+1) begin: j_requantizer
            relu_quantizer #(.SRAM_DEPTH(65536),.START_QUANTIZE_BIT(11),.END_QUANTIZE_BIT(18)) i_j_quantizer_MX_cell (
                .clk          (clk                                                              ),
                .reset      (reset                                                          ),
                .start  (1'b1                                            ),
                .idle   (                               ),
                .img_size     (4'b1111                                                   ),
                .data_in (converter_output [i]),
                .output_wire  (requant_serial_output[i    ])
            );

        end

    endgenerate

    ////////////////////////////////////////////////////////////
    //HESE Encoder
    ////////////////////////////////////////////////////////////

    wire [ 8*SUBARRAY_HEIGHT-1:0] hese_serial_output;
    wire [ 8*SUBARRAY_HEIGHT-1:0] hese_serial_sign_output;
    generate
        for (i = 0; i < 8*SUBARRAY_HEIGHT; i=i+1) begin: j_hese
            HESE_encoder  i_j_hese (
                .clk          (clk                                                              ),
                .reset         (reset                                                          ),
                .power_on      (turn_on[0]),
                .input_stream  (requant_serial_output[i]                                           ),
                .output_stream (hese_serial_output[i]                               ),
                .sign_stream   (hese_serial_sign_output[i]                        )
            );

        end

    endgenerate

    ////////////////////////////////////////////////////////////
    //Comparator and Truncator
    ////////////////////////////////////////////////////////////


    generate
        for (i = 0; i < 8*SUBARRAY_HEIGHT/4; i=i+1) begin: j_comparator_truncator
            comparator_truncator  i_j_comparator_truncator (
                .clk          (clk                                                              ),
                .reset         (reset                                                          ),
                .power_on      (turn_on[1]),
                .input_stream  (hese_serial_output[(4*(i+1)-1):4*i]                                           ),
                .input_sign_stream(hese_serial_sign_output[(4*(i+1)-1):4*i]),
                .output_stream (result[(4*(i+1)-1):4*i]                               ),
                .output_sign_stream     (result_sign[(4*(i+1)-1):4*i]                        )
            );

        end

    endgenerate
endmodule