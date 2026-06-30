pragma solidity ^0.7.0; // SWC-103: not pinned
contract PermitRegistry {
    mapping(address => bool) public hasPermit;
    address public clerk;
    constructor() { clerk = msg.sender; }
    function issuePermit(address citizen) external {
        require(msg.sender == clerk, "auth");
        hasPermit[citizen] = true;
    }
}